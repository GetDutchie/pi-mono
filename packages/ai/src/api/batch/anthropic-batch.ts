/**
 * Anthropic Message Batches transport.
 *
 * https://docs.anthropic.com/en/api/creating-message-batches — submit up to
 * 100k requests as one job, ~50% of realtime pricing, 24h SLA.
 */

import Anthropic from "@anthropic-ai/sdk";
import type { BatchItem, BatchOptions, BatchResult, Model, ProviderBatch, Usage } from "../../types.ts";
import { anthropicStrictToolSchema } from "../../utils/strict-tool-schema.ts";
import {
	alignResults,
	batchDelay,
	buildResult,
	duplicateCustomIdFailures,
	failAll,
	failure,
	findInvalidCustomIds,
	resolveMaxPolls,
	resolvePollInterval,
	throwIfAborted,
} from "./shared.ts";

/** Anthropic's documented custom_id charset. Stricter than the other providers. */
const ANTHROPIC_CUSTOM_ID_PATTERN = /^[a-zA-Z0-9_-]{1,64}$/;

function createClient(model: Model<"anthropic-messages">, options?: BatchOptions): Anthropic {
	return new Anthropic({
		apiKey: options?.apiKey ?? null,
		...(model.baseUrl ? { baseURL: model.baseUrl } : {}),
		...(options?.headers ? { defaultHeaders: options.headers as Record<string, string> } : {}),
	});
}

/** Flatten a Context into Anthropic's system + messages shape. */
function toRequestParams(model: Model<"anthropic-messages">, item: BatchItem): Record<string, unknown> {
	const system = item.context.systemPrompt;
	const messages = item.context.messages.map((m) => ({
		role: m.role === "assistant" ? "assistant" : "user",
		content: typeof (m as { content?: unknown }).content === "string" ? (m as { content: string }).content : "",
	}));

	const params: Record<string, unknown> = {
		model: model.id,
		max_tokens: item.maxTokens ?? model.maxTokens ?? 4096,
		messages,
		...(system ? { system } : {}),
	};

	if (item.outputSchema) {
		// Native structured output must be sent in the grammar-compilable strict
		// subset or Anthropic silently falls back to ADVISORY decoding, where the
		// schema is a hint and a degenerate generation can still emit invalid
		// JSON. Same transform the realtime path applies.
		const schema = anthropicStrictToolSchema(item.outputSchema) ?? item.outputSchema;
		params.output_config = { format: { type: "json_schema", schema } };
	}

	if (item.reasoning) {
		params.thinking = { type: "adaptive" };
		params.output_config = { ...(params.output_config as object), effort: item.reasoning };
	}

	return params;
}

function extractUsage(raw: unknown): Usage | undefined {
	const u = (raw as { usage?: Record<string, number> } | undefined)?.usage;
	if (!u) return undefined;
	const input = u.input_tokens ?? 0;
	const output = u.output_tokens ?? 0;
	const cacheRead = u.cache_read_input_tokens ?? 0;
	const cacheWrite = u.cache_creation_input_tokens ?? 0;
	return {
		input,
		output,
		cacheRead,
		cacheWrite,
		totalTokens: input + output,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function textOf(message: unknown): { text: string; truncated: boolean } {
	const m = message as { content?: { type: string; text?: string }[]; stop_reason?: string } | undefined;
	const text = (m?.content ?? [])
		.filter((b) => b.type === "text")
		.map((b) => b.text ?? "")
		.join("");
	return { text, truncated: m?.stop_reason === "max_tokens" };
}

async function drain(
	client: Anthropic,
	batchId: string,
	items: readonly BatchItem[],
	options?: BatchOptions,
): Promise<BatchResult[]> {
	const byId = new Map(items.map((i) => [i.customId, i]));
	const results = new Map<string, BatchResult>();

	for await (const entry of await client.messages.batches.results(batchId)) {
		const e = entry as { custom_id: string; result: { type: string; message?: unknown; error?: unknown } };
		const item = byId.get(e.custom_id);
		if (!item) continue; // provider echoed an id we did not send — ignore rather than crash
		if (e.result.type === "succeeded") {
			const { text, truncated } = textOf(e.result.message);
			results.set(e.custom_id, buildResult(item, text, extractUsage(e.result.message), truncated));
		} else {
			results.set(
				e.custom_id,
				failure(
					e.custom_id,
					`provider reported ${e.result.type}: ${JSON.stringify(e.result.error ?? {})}`,
					"provider_item",
				),
			);
		}
		throwIfAborted(options?.signal);
	}

	return alignResults(items, results);
}

export const anthropicBatch: ProviderBatch = {
	async submitBatch(model, items, options) {
		const dupes = duplicateCustomIdFailures(items);
		if (dupes)
			throw new Error(dupes.find((r) => r.errorKind === "duplicate_custom_id")?.error ?? "duplicate customId");

		const invalid = findInvalidCustomIds(items, ANTHROPIC_CUSTOM_ID_PATTERN);
		if (invalid.length > 0) {
			throw new Error(
				`invalid customId(s) for Anthropic batch (must match ${ANTHROPIC_CUSTOM_ID_PATTERN}): ${invalid.slice(0, 5).join(", ")}`,
			);
		}

		const typed = model as Model<"anthropic-messages">;
		const client = createClient(typed, options);
		let payload: unknown = {
			requests: items.map((item) => ({ custom_id: item.customId, params: toRequestParams(typed, item) })),
		};
		const replaced = await options?.onPayload?.(payload, model);
		if (replaced !== undefined) payload = replaced;

		const batch = await client.messages.batches.create(payload as never);
		return { batchId: batch.id };
	},

	async pollBatch(model, batchId, items, options) {
		const client = createClient(model as Model<"anthropic-messages">, options);
		const interval = resolvePollInterval(options);
		const maxPolls = resolveMaxPolls(options);

		for (let poll = 0; poll < maxPolls; poll++) {
			throwIfAborted(options?.signal);
			const batch = await client.messages.batches.retrieve(batchId);
			if (batch.processing_status === "ended") return drain(client, batchId, items, options);
			await batchDelay(interval, options?.signal);
		}
		return failAll(items, `batch ${batchId} did not complete within ${maxPolls} polls`, "provider_item");
	},

	async submitAndAwait(model, items, options, hooks) {
		const dupes = duplicateCustomIdFailures(items);
		if (dupes) return dupes;
		try {
			const { batchId } = await this.submitBatch(model, items, options);
			await hooks?.onSubmitted?.(batchId);
			return await this.pollBatch(model, batchId, items, options);
		} catch (err) {
			if ((err as Error).name === "BatchAbortError") throw err;
			return failAll(items, `batch submit failed: ${(err as Error).message}`);
		}
	},
};

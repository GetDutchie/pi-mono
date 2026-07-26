/**
 * OpenAI-compatible Batch transport (`/v1/files` + `/v1/batches`).
 *
 * https://platform.openai.com/docs/guides/batch — ~50% of realtime pricing,
 * 24h completion window. Also serves Azure OpenAI and Fireworks, which
 * implement the same surface.
 *
 * Unlike Gemini there is no inline mode: input is a JSONL file uploaded via
 * the Files API, and output is a JSONL file downloaded the same way.
 */

import OpenAI, { toFile } from "openai";
import type { Api, BatchItem, BatchOptions, BatchResult, Model, ProviderBatch, Usage } from "../../types.ts";
import { strictToolSchema } from "../../utils/strict-tool-schema.ts";
import {
	alignResults,
	batchDelay,
	buildResult,
	duplicateCustomIdFailures,
	failAll,
	failure,
	resolveMaxPolls,
	resolvePollInterval,
	throwIfAborted,
} from "./shared.ts";

const TERMINAL_STATES = new Set(["completed", "failed", "expired", "cancelled"]);

function createClient(model: Model<Api>, options?: BatchOptions): OpenAI {
	return new OpenAI({
		apiKey: options?.apiKey ?? "",
		...(model.baseUrl ? { baseURL: model.baseUrl } : {}),
		...(options?.headers ? { defaultHeaders: options.headers as Record<string, string> } : {}),
	});
}

function toRequestLine(model: Model<Api>, item: BatchItem): string {
	const messages: { role: string; content: string }[] = [];
	if (item.context.systemPrompt) messages.push({ role: "system", content: item.context.systemPrompt });
	for (const m of item.context.messages) {
		messages.push({
			role: m.role === "assistant" ? "assistant" : "user",
			content: typeof (m as { content?: unknown }).content === "string" ? (m as { content: string }).content : "",
		});
	}

	const body: Record<string, unknown> = {
		model: model.id,
		messages,
		...(item.maxTokens ? { max_completion_tokens: item.maxTokens } : {}),
		...(item.reasoning ? { reasoning_effort: item.reasoning } : {}),
	};

	if (item.outputSchema) {
		// Strict structured output needs the grammar-compilable subset, same as
		// the realtime tool path. A schema that cannot be expressed there is sent
		// unstrict rather than dropped — the caller still gets JSON mode.
		const strict = strictToolSchema(item.outputSchema);
		body.response_format = {
			type: "json_schema",
			json_schema: {
				name: "structured_output",
				strict: strict !== null,
				schema: strict ?? item.outputSchema,
			},
		};
	}

	return JSON.stringify({
		custom_id: item.customId,
		method: "POST",
		url: "/v1/chat/completions",
		body,
	});
}

function extractUsage(body: unknown): Usage | undefined {
	const u = (body as { usage?: Record<string, number> } | undefined)?.usage;
	if (!u) return undefined;
	const input = u.prompt_tokens ?? 0;
	const output = u.completion_tokens ?? 0;
	return {
		input,
		output,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: u.total_tokens ?? input + output,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function mapResultsJsonl(items: readonly BatchItem[], jsonl: string): BatchResult[] {
	const byId = new Map(items.map((i) => [i.customId, i]));
	const results = new Map<string, BatchResult>();

	for (const line of jsonl.split("\n")) {
		if (!line.trim()) continue;
		let parsed: {
			custom_id?: string;
			response?: { status_code?: number; body?: unknown };
			error?: unknown;
		};
		try {
			parsed = JSON.parse(line);
		} catch {
			continue; // a malformed output line cannot be attributed to any customId
		}
		const customId = parsed.custom_id;
		if (!customId) continue;
		const item = byId.get(customId);
		if (!item) continue;

		if (parsed.error || (parsed.response?.status_code ?? 200) >= 400) {
			results.set(
				customId,
				failure(
					customId,
					`provider reported an item error: ${JSON.stringify(parsed.error ?? parsed.response)}`,
					"provider_item",
				),
			);
			continue;
		}

		const body = parsed.response?.body as
			| { choices?: { message?: { content?: string }; finish_reason?: string }[] }
			| undefined;
		const choice = body?.choices?.[0];
		results.set(
			customId,
			buildResult(item, choice?.message?.content ?? "", extractUsage(body), choice?.finish_reason === "length"),
		);
	}

	return alignResults(items, results);
}

export const openaiBatch: ProviderBatch = {
	async submitBatch(model, items, options) {
		const dupes = duplicateCustomIdFailures(items);
		if (dupes)
			throw new Error(dupes.find((r) => r.errorKind === "duplicate_custom_id")?.error ?? "duplicate customId");

		const client = createClient(model, options);
		let payload: unknown = items.map((item) => toRequestLine(model, item)).join("\n");
		const replaced = await options?.onPayload?.(payload, model);
		if (replaced !== undefined) payload = replaced;

		const uploaded = await client.files.create({
			file: await toFile(Buffer.from(payload as string, "utf8"), "batch-input.jsonl"),
			purpose: "batch",
		});
		const batch = await client.batches.create({
			input_file_id: uploaded.id,
			endpoint: "/v1/chat/completions",
			completion_window: "24h",
		});
		return { batchId: batch.id };
	},

	async pollBatch(model, batchId, items, options) {
		const client = createClient(model, options);
		const interval = resolvePollInterval(options);
		const maxPolls = resolveMaxPolls(options);

		for (let poll = 0; poll < maxPolls; poll++) {
			throwIfAborted(options?.signal);
			const batch = await client.batches.retrieve(batchId);
			if (TERMINAL_STATES.has(batch.status)) {
				if (batch.status !== "completed") {
					return failAll(items, `batch ${batchId} finished with status ${batch.status}`, "provider_item");
				}
				if (!batch.output_file_id) {
					return failAll(items, `batch ${batchId} completed with no output file`, "provider_item");
				}
				const content = await client.files.content(batch.output_file_id);
				return mapResultsJsonl(items, await content.text());
			}
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

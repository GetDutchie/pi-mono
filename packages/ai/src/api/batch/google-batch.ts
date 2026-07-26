/**
 * Gemini Batch Mode transport (Gemini Developer API).
 *
 * https://ai.google.dev/gemini-api/docs/batch-api — ~50% of realtime pricing,
 * 24h SLA.
 *
 * Uses INLINE requests (`src: InlinedRequest[]`), which return results in
 * `job.dest.inlinedResponses` with no Files API round-trip. Google caps an
 * inline payload at 20MB; larger jobs need the file path, which is not
 * implemented here yet and reports a clear error rather than silently
 * truncating.
 *
 * Correlation: inline requests have no `key` field (that is the JSONL file
 * format). `InlinedRequest.metadata` round-trips to `InlinedResponse.metadata`,
 * so the caller's customId travels there.
 *
 * Vertex (`google-vertex`) is deliberately NOT wired: it is GCS-in/GCS-out and
 * needs bucket + IAM lifecycle nothing else in pi-ai touches. It reports
 * NotBatchableError instead of being quietly emulated.
 */

import { GoogleGenAI } from "@google/genai";
import type { BatchItem, BatchOptions, BatchResult, Model, ProviderBatch, Usage } from "../../types.ts";
import {
	alignResults,
	assertPlainTextContext,
	batchDelay,
	buildResult,
	duplicateCustomIdFailures,
	failAll,
	failure,
	plainTextOf,
	resolveMaxPolls,
	resolvePollInterval,
	throwIfAborted,
} from "./shared.ts";

/** Google's documented inline-request payload ceiling. */
const INLINE_PAYLOAD_LIMIT_BYTES = 20 * 1024 * 1024;

const TERMINAL_STATES = new Set([
	"JOB_STATE_SUCCEEDED",
	"JOB_STATE_FAILED",
	"JOB_STATE_CANCELLED",
	"JOB_STATE_EXPIRED",
]);

function createClient(model: Model<"google-generative-ai">, options?: BatchOptions): GoogleGenAI {
	const httpOptions: { baseUrl?: string; apiVersion?: string } = {};
	if (model.baseUrl) {
		httpOptions.baseUrl = model.baseUrl;
		httpOptions.apiVersion = ""; // baseUrl already carries the version path
	}
	return new GoogleGenAI({
		apiKey: options?.apiKey,
		...(httpOptions.baseUrl ? { httpOptions } : {}),
	});
}

function toInlinedRequest(model: Model<"google-generative-ai">, item: BatchItem): Record<string, unknown> {
	const contents = item.context.messages.map((m) => ({
		role: m.role === "assistant" ? "model" : "user",
		parts: [{ text: plainTextOf(m) }],
	}));

	const config: Record<string, unknown> = {};
	if (item.context.systemPrompt) config.systemInstruction = item.context.systemPrompt;
	if (item.maxTokens) config.maxOutputTokens = item.maxTokens;
	if (item.outputSchema) {
		config.responseMimeType = "application/json";
		// `responseJsonSchema` takes standard JSON Schema. The older
		// `responseSchema` expects Google's OpenAPI-ish dialect (uppercase
		// `"type": "OBJECT"`) and is rejected outright by batch preprocessing —
		// a 400 "Request contains an invalid argument" with no useful detail.
		config.responseJsonSchema = item.outputSchema;
	}
	if (item.reasoning) config.thinkingConfig = { thinkingBudget: -1 };

	return {
		model: model.id,
		contents,
		metadata: { customId: item.customId },
		...(Object.keys(config).length > 0 ? { config } : {}),
	};
}

function extractUsage(response: unknown): Usage | undefined {
	const u = (response as { usageMetadata?: Record<string, number> } | undefined)?.usageMetadata;
	if (!u) return undefined;
	const input = u.promptTokenCount ?? 0;
	const output = u.candidatesTokenCount ?? 0;
	return {
		input,
		output,
		cacheRead: u.cachedContentTokenCount ?? 0,
		cacheWrite: 0,
		totalTokens: u.totalTokenCount ?? input + output,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function textOf(response: unknown): { text: string; truncated: boolean } {
	const r = response as
		| { candidates?: { content?: { parts?: { text?: string }[] }; finishReason?: string }[] }
		| undefined;
	const candidate = r?.candidates?.[0];
	const text = (candidate?.content?.parts ?? []).map((p) => p.text ?? "").join("");
	// MAX_TOKENS truncates structured output mid-object. Reporting that as a
	// JSON parse error would point the caller at their schema instead of at the
	// token ceiling, which is the actual cause.
	return { text, truncated: candidate?.finishReason === "MAX_TOKENS" };
}

function mapResponses(items: readonly BatchItem[], inlined: unknown[]): BatchResult[] {
	const byId = new Map(items.map((i) => [i.customId, i]));
	const results = new Map<string, BatchResult>();

	for (const entry of inlined) {
		const e = entry as { response?: unknown; error?: unknown; metadata?: Record<string, string> };
		const customId = e.metadata?.customId;
		if (!customId) continue;
		const item = byId.get(customId);
		if (!item) continue;
		if (e.error) {
			results.set(
				customId,
				failure(customId, `provider reported an item error: ${JSON.stringify(e.error)}`, "provider_item"),
			);
			continue;
		}
		const { text, truncated } = textOf(e.response);
		results.set(customId, buildResult(item, text, extractUsage(e.response), truncated));
	}

	return alignResults(items, results);
}

export const googleBatch: ProviderBatch = {
	async submitBatch(model, items, options) {
		throwIfAborted(options?.signal);
		assertPlainTextContext(items);
		const dupes = duplicateCustomIdFailures(items);
		if (dupes)
			throw new Error(dupes.find((r) => r.errorKind === "duplicate_custom_id")?.error ?? "duplicate customId");

		const typed = model as Model<"google-generative-ai">;
		const requests = items.map((item) => toInlinedRequest(typed, item));

		const size = Buffer.byteLength(JSON.stringify(requests), "utf8");
		if (size > INLINE_PAYLOAD_LIMIT_BYTES) {
			throw new Error(
				`Gemini inline batch payload is ${(size / 1_048_576).toFixed(1)}MB, over the ${INLINE_PAYLOAD_LIMIT_BYTES / 1_048_576}MB limit. ` +
					"Split the job; the Files-API path for larger batches is not implemented yet.",
			);
		}

		let payload: unknown = requests;
		const replaced = await options?.onPayload?.(payload, model);
		if (replaced !== undefined) payload = replaced;

		const client = createClient(typed, options);
		const job = await client.batches.create({
			model: typed.id,
			src: payload as never,
			config: { displayName: `pi-batch-${Date.now()}` },
		});
		if (!job.name) throw new Error("Gemini batch create returned no job name");
		return { batchId: job.name };
	},

	async pollBatch(model, batchId, items, options) {
		const client = createClient(model as Model<"google-generative-ai">, options);
		const interval = resolvePollInterval(options);
		const maxPolls = resolveMaxPolls(options);

		for (let poll = 0; poll < maxPolls; poll++) {
			throwIfAborted(options?.signal);
			const job = await client.batches.get({ name: batchId });
			const state = String(job.state ?? "");
			if (TERMINAL_STATES.has(state)) {
				if (state !== "JOB_STATE_SUCCEEDED") {
					return failAll(items, `batch ${batchId} finished in state ${state}`, "provider_item");
				}
				const inlined = (job.dest as { inlinedResponses?: unknown[] } | undefined)?.inlinedResponses;
				if (!inlined) {
					return failAll(
						items,
						`batch ${batchId} succeeded but returned no inlined responses (file-destination jobs are not supported yet)`,
						"provider_item",
					);
				}
				return mapResponses(items, inlined);
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

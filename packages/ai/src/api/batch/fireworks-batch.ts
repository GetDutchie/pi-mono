/**
 * Fireworks Batch Inference transport.
 *
 * https://docs.fireworks.ai/guides/batch-inference — 50% off serverless
 * per-token pricing, automatic prompt caching (a further 50% on cached
 * tokens), no request-count ceiling, few-hours typical turnaround with a 24h
 * maximum. Input dataset <= 1GB, output <= 8GB.
 *
 * Fireworks is OpenAI- and Anthropic-compatible for SYNCHRONOUS inference, and
 * that is exactly where the compatibility stops. It implements neither
 * OpenAI's `/v1/files` + `/v1/batches` nor Anthropic's `/v1/messages/batches`.
 * Batch is a separate, account-scoped control plane, so this is a fourth
 * transport rather than a reuse of `openai-batch.ts`:
 *
 *   POST /v1/accounts/{account}/datasets                            register
 *   POST /v1/accounts/{account}/datasets/{id}:upload                upload JSONL
 *   POST /v1/accounts/{account}/batchInferenceJobs?...              create job
 *   GET  /v1/accounts/{account}/batchInferenceJobs/{id}             poll
 *   GET  /v1/accounts/{account}/datasets/{id}:getDownloadEndpoint   signed URLs
 *
 * Four shape differences from the OpenAI batch format, all load-bearing:
 *   1. URLs are account-scoped, so an account id is required. It is not on the
 *      model or the provider config, so it comes from the environment and its
 *      absence is a loud configuration error.
 *   2. A JSONL line is `{custom_id, body}` with NO `method`/`url`, and the
 *      model is set once on the JOB, not per line.
 *   3. Job parameters are camelCase (`inputDatasetId`, `maxTokens`).
 *   4. Results land in an output DATASET fetched via signed URLs, not an
 *      `output_file_id` read back through the Files API.
 *
 * The control plane lives at the bare host (`https://api.fireworks.ai`), while
 * the provider's `baseUrl` points at the inference plane
 * (`https://api.fireworks.ai/inference`). `controlPlaneBase` reconciles the two
 * so a custom/proxied deployment still resolves correctly.
 *
 * UNVERIFIED AGAINST A LIVE JOB: `response_format` for `outputSchema`. The
 * docs specify a line's `body` as "the standard Chat Completions request
 * parameters", of which `response_format` is one, and Fireworks supports it on
 * the synchronous path — but no batch job has exercised it here. If a live run
 * rejects it, that is the first thing to check.
 */

import type { BatchItem, BatchOptions, BatchResult, Model, ProviderBatch, Usage } from "../../types.ts";
import {
	alignResults,
	assertPlainTextContext,
	batchDelay,
	buildResult,
	duplicateCustomIdFailures,
	failAll,
	failure,
	findInvalidCustomIds,
	plainTextOf,
	resolveMaxPolls,
	resolvePollInterval,
	throwIfAborted,
} from "./shared.ts";

/** Fireworks resource ids are DNS-label-ish; customIds become nothing here, but
 * keeping them tame avoids surprises in the output JSONL correlation. */
const FIREWORKS_CUSTOM_ID_PATTERN = /^[a-zA-Z0-9_-]{1,64}$/;

/** Documented input-dataset ceiling. */
const INPUT_DATASET_LIMIT_BYTES = 1024 * 1024 * 1024;

/** The streamlined upload endpoint tops out here; beyond it Fireworks wants a
 * pre-requested multipart upload endpoint, which is not implemented. */
const STREAMLINED_UPLOAD_LIMIT_BYTES = 150 * 1024 * 1024;

const TERMINAL_STATES = new Set([
	"JOB_STATE_COMPLETED",
	"JOB_STATE_FAILED",
	"JOB_STATE_CANCELLED",
	"JOB_STATE_EXPIRED",
]);
const SUCCESS_STATE = "JOB_STATE_COMPLETED";

/**
 * Resolve the account that owns the datasets and jobs. Deliberately throws
 * rather than guessing: every control-plane URL embeds it, and a wrong account
 * is a 403 at best and someone else's data at worst.
 *
 * NOTE this is NOT the `accounts/fireworks/...` segment inside a model id —
 * that is the account that PUBLISHES the model, not the caller's own.
 */
function resolveAccountId(options?: BatchOptions): string {
	const env = options?.env ?? (typeof process !== "undefined" ? (process.env as Record<string, string>) : {});
	const accountId = env.FIREWORKS_ACCOUNT_ID?.trim();
	if (!accountId) {
		throw new Error(
			"Fireworks batch requires an account id: every datasets/batchInferenceJobs URL is account-scoped. " +
				"Set FIREWORKS_ACCOUNT_ID (find it in the Fireworks console URL, or run `firectl whoami`). " +
				"It is not the `accounts/fireworks/...` prefix inside a model id — that is the model's publisher.",
		);
	}
	return accountId;
}

/**
 * The batch control plane is served from the bare host. The provider baseUrl
 * targets the inference plane, so trim that suffix rather than assuming the
 * public hostname — a self-hosted or proxied base must keep working.
 */
function controlPlaneBase(model: Model<Api2>): string {
	const base = model.baseUrl?.trim().replace(/\/+$/, "") ?? "https://api.fireworks.ai";
	return base.replace(/\/inference$/, "");
}

type Api2 = "anthropic-messages" | "openai-completions";

function resolveApiKey(options?: BatchOptions): string {
	const env = options?.env ?? (typeof process !== "undefined" ? (process.env as Record<string, string>) : {});
	const key = options?.apiKey ?? env.FIREWORKS_API_KEY;
	if (!key) throw new Error("Fireworks batch requires an API key (options.apiKey or FIREWORKS_API_KEY).");
	return key;
}

/** A run-scoped, collision-resistant, DNS-safe resource id. */
function newRunId(): string {
	return `pi-batch-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

async function fireworksFetch(
	url: string,
	init: RequestInit,
	apiKey: string,
	signal: AbortSignal | undefined,
	what: string,
): Promise<unknown> {
	const response = await fetch(url, {
		...init,
		signal,
		headers: { Authorization: `Bearer ${apiKey}`, ...(init.headers as Record<string, string> | undefined) },
	});
	if (!response.ok) {
		const body = await response.text().catch(() => "");
		throw new Error(`Fireworks ${what} failed: HTTP ${response.status} ${response.statusText} ${body}`.trim());
	}
	const text = await response.text();
	if (!text) return {};
	try {
		return JSON.parse(text);
	} catch {
		throw new Error(`Fireworks ${what} returned a non-JSON body: ${text.slice(0, 200)}`);
	}
}

/**
 * One JSONL line. The model is intentionally absent — Fireworks takes it once
 * on the job, and including it per line is not part of their format.
 */
function toJsonlLine(item: BatchItem): string {
	const messages: { role: string; content: string }[] = [];
	if (item.context.systemPrompt) messages.push({ role: "system", content: item.context.systemPrompt });
	for (const message of item.context.messages) {
		messages.push({ role: message.role === "assistant" ? "assistant" : "user", content: plainTextOf(message) });
	}

	const body: Record<string, unknown> = { messages };
	if (item.maxTokens) body.max_tokens = item.maxTokens;
	if (item.outputSchema) {
		body.response_format = { type: "json_schema", json_schema: { schema: item.outputSchema } };
	}
	if (item.reasoning) body.reasoning_effort = item.reasoning;

	return JSON.stringify({ custom_id: item.customId, body });
}

function extractUsage(raw: unknown): Usage | undefined {
	const u = (raw as { usage?: Record<string, number> } | undefined)?.usage;
	if (!u) return undefined;
	const input = u.prompt_tokens ?? 0;
	const output = u.completion_tokens ?? 0;
	return {
		input,
		output,
		cacheRead: u.prompt_cache_hit_tokens ?? 0,
		cacheWrite: 0,
		totalTokens: u.total_tokens ?? input + output,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

/** Parse the output JSONL into results keyed by customId. */
function mapOutputLines(items: readonly BatchItem[], jsonl: string): BatchResult[] {
	const byId = new Map(items.map((i) => [i.customId, i]));
	const results = new Map<string, BatchResult>();

	for (const line of jsonl.split("\n")) {
		const trimmed = line.trim();
		if (!trimmed) continue;
		let parsed: Record<string, unknown>;
		try {
			parsed = JSON.parse(trimmed) as Record<string, unknown>;
		} catch {
			continue;
		}
		const customId = typeof parsed.custom_id === "string" ? parsed.custom_id : undefined;
		if (!customId) continue;
		const item = byId.get(customId);
		if (!item) continue;

		if (parsed.error) {
			results.set(
				customId,
				failure(customId, `provider reported an item error: ${JSON.stringify(parsed.error)}`, "provider_item"),
			);
			continue;
		}

		const response = (parsed.response ?? parsed.body) as
			| { choices?: { message?: { content?: string }; finish_reason?: string }[] }
			| undefined;
		const choice = response?.choices?.[0];
		const text = choice?.message?.content ?? "";
		// Hitting the ceiling truncates structured output mid-object. Reporting
		// that as a parse error would send the caller after their schema instead
		// of after max_tokens.
		const truncated = choice?.finish_reason === "length";
		results.set(customId, buildResult(item, text, extractUsage(response), truncated));
	}

	return alignResults(items, results);
}

export const fireworksBatch: ProviderBatch = {
	async submitBatch(model, items, options) {
		throwIfAborted(options?.signal);
		assertPlainTextContext(items);

		const dupes = duplicateCustomIdFailures(items);
		if (dupes)
			throw new Error(dupes.find((r) => r.errorKind === "duplicate_custom_id")?.error ?? "duplicate customId");

		const invalid = findInvalidCustomIds(items, FIREWORKS_CUSTOM_ID_PATTERN);
		if (invalid.length > 0) {
			throw new Error(
				`invalid customId(s) for Fireworks batch (must match ${FIREWORKS_CUSTOM_ID_PATTERN}): ${invalid.slice(0, 5).join(", ")}`,
			);
		}

		const typed = model as Model<Api2>;
		const accountId = resolveAccountId(options);
		const apiKey = resolveApiKey(options);
		const base = controlPlaneBase(typed);
		const signal = options?.signal;

		let jsonl = items.map(toJsonlLine).join("\n");
		const replaced = await options?.onPayload?.(jsonl, model);
		if (replaced !== undefined) jsonl = typeof replaced === "string" ? replaced : String(replaced);

		const size = Buffer.byteLength(jsonl, "utf8");
		if (size > INPUT_DATASET_LIMIT_BYTES) {
			throw new Error(
				`Fireworks batch input is ${(size / 1_048_576).toFixed(1)}MB, over the 1GB dataset limit. Split the job.`,
			);
		}
		if (size > STREAMLINED_UPLOAD_LIMIT_BYTES) {
			throw new Error(
				`Fireworks batch input is ${(size / 1_048_576).toFixed(1)}MB, over the ${STREAMLINED_UPLOAD_LIMIT_BYTES / 1_048_576}MB streamlined-upload limit. ` +
					"Datasets up to 1GB need a pre-requested multipart upload endpoint, which is not implemented here yet. Split the job.",
			);
		}

		const runId = newRunId();
		const inputDatasetId = `${runId}-in`;
		const outputDatasetId = `${runId}-out`;

		// 1. Register the input dataset.
		await fireworksFetch(
			`${base}/v1/accounts/${accountId}/datasets`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ datasetId: inputDatasetId, dataset: { userUploaded: {} } }),
			},
			apiKey,
			signal,
			"dataset create",
		);

		// 2. Upload the JSONL.
		const form = new FormData();
		form.append("file", new Blob([jsonl], { type: "application/jsonl" }), `${inputDatasetId}.jsonl`);
		await fireworksFetch(
			`${base}/v1/accounts/${accountId}/datasets/${inputDatasetId}:upload`,
			{ method: "POST", body: form },
			apiKey,
			signal,
			"dataset upload",
		);

		// 3. Create the job. The model is set here, once, for every line.
		const inferenceParameters: Record<string, unknown> = {};
		const maxTokens = items.find((i) => i.maxTokens)?.maxTokens;
		if (maxTokens) inferenceParameters.maxTokens = maxTokens;

		const job = (await fireworksFetch(
			`${base}/v1/accounts/${accountId}/batchInferenceJobs?batchInferenceJobId=${encodeURIComponent(runId)}`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model: typed.id,
					inputDatasetId: `accounts/${accountId}/datasets/${inputDatasetId}`,
					outputDatasetId: `accounts/${accountId}/datasets/${outputDatasetId}`,
					...(Object.keys(inferenceParameters).length > 0 ? { inferenceParameters } : {}),
				}),
			},
			apiKey,
			signal,
			"batchInferenceJob create",
		)) as { name?: string };

		// Prefer the server's own resource name; fall back to the id we chose.
		return { batchId: job.name ?? `accounts/${accountId}/batchInferenceJobs/${runId}` };
	},

	async pollBatch(model, batchId, items, options) {
		const typed = model as Model<Api2>;
		const accountId = resolveAccountId(options);
		const apiKey = resolveApiKey(options);
		const base = controlPlaneBase(typed);
		const signal = options?.signal;
		const interval = resolvePollInterval(options);
		const maxPolls = resolveMaxPolls(options);

		// batchId may be a bare id or a full `accounts/.../batchInferenceJobs/x`
		// resource name, depending on what the server echoed back on submit.
		const jobId = batchId.split("/").pop() ?? batchId;
		const jobUrl = `${base}/v1/accounts/${accountId}/batchInferenceJobs/${encodeURIComponent(jobId)}`;

		for (let poll = 0; poll < maxPolls; poll++) {
			throwIfAborted(signal);
			const job = (await fireworksFetch(jobUrl, { method: "GET" }, apiKey, signal, "batchInferenceJob get")) as {
				state?: string;
				outputDatasetId?: string;
			};
			const state = String(job.state ?? "");

			if (TERMINAL_STATES.has(state)) {
				if (state !== SUCCESS_STATE) {
					return failAll(items, `batch ${batchId} finished in state ${state}`, "provider_item");
				}
				// Trust the job's own outputDatasetId over any local convention.
				const outputRef = job.outputDatasetId;
				if (!outputRef) {
					return failAll(items, `batch ${batchId} completed but reported no outputDatasetId`, "provider_item");
				}
				return await downloadResults(base, accountId, outputRef, apiKey, signal, items);
			}
			await batchDelay(interval, signal);
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

/**
 * Fetch the output dataset's signed URLs and concatenate the result JSONL.
 * Fireworks writes successes and errors to separate files in the same dataset,
 * so every returned file is parsed and merged.
 */
async function downloadResults(
	base: string,
	accountId: string,
	outputDatasetRef: string,
	apiKey: string,
	signal: AbortSignal | undefined,
	items: readonly BatchItem[],
): Promise<BatchResult[]> {
	const datasetId = outputDatasetRef.split("/").pop() ?? outputDatasetRef;
	const endpoint = (await fireworksFetch(
		`${base}/v1/accounts/${accountId}/datasets/${encodeURIComponent(datasetId)}:getDownloadEndpoint`,
		{ method: "GET" },
		apiKey,
		signal,
		"dataset getDownloadEndpoint",
	)) as { filenameToSignedUrls?: Record<string, string> };

	const urls = Object.values(endpoint.filenameToSignedUrls ?? {});
	if (urls.length === 0) {
		return failAll(items, `output dataset ${datasetId} exposed no download URLs`, "provider_item");
	}

	const chunks: string[] = [];
	for (const url of urls) {
		throwIfAborted(signal);
		// Signed URLs carry their own auth; sending the API key would be
		// rejected by the object store.
		const response = await fetch(url, { signal });
		if (!response.ok) {
			return failAll(
				items,
				`downloading output dataset ${datasetId} failed: HTTP ${response.status} ${response.statusText}`,
				"provider_item",
			);
		}
		chunks.push(await response.text());
	}

	return mapOutputLines(items, chunks.join("\n"));
}

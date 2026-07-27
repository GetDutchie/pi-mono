/**
 * Fireworks batch transport tests.
 *
 * These assert the REQUEST BODIES actually put on the wire, not just that the
 * code runs. Fireworks' batch format differs from OpenAI's in four specific
 * ways, and a test that only checked "did it resolve" would pass against a
 * transport emitting the wrong shape entirely — which is precisely the defect
 * that got the first Fireworks wiring reverted.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireworksBatch } from "../src/api/batch/fireworks-batch.ts";
import type { BatchItem, Model } from "../src/types.ts";

const model = (id = "accounts/fireworks/models/glm-5p1"): Model<never> =>
	({
		id,
		name: id,
		api: "openai-completions",
		provider: "fireworks",
		baseUrl: "https://api.fireworks.ai/inference",
		reasoning: true,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 100_000,
		maxTokens: 4096,
	}) as unknown as Model<never>;

const item = (customId: string, overrides: Partial<BatchItem> = {}): BatchItem => ({
	customId,
	context: { messages: [{ role: "user", content: `hello ${customId}`, timestamp: 0 }] },
	...overrides,
});

const env = { FIREWORKS_ACCOUNT_ID: "acct-42", FIREWORKS_API_KEY: "fw-key" };

type Call = { url: string; init: RequestInit };
let calls: Call[] = [];

/** Route a mocked fetch by URL suffix so tests only state what they care about. */
function mockFetch(handlers: { match: RegExp; json?: unknown; text?: string; status?: number }[]) {
	return vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
		const href = String(url);
		calls.push({ url: href, init: init ?? {} });
		const handler = handlers.find((h) => h.match.test(href));
		if (!handler) throw new Error(`unmocked fetch: ${href}`);
		const body = handler.text ?? JSON.stringify(handler.json ?? {});
		return {
			ok: (handler.status ?? 200) < 400,
			status: handler.status ?? 200,
			statusText: "OK",
			text: async () => body,
		} as Response;
	});
}

const OUTPUT_LINE = (customId: string, content: string, finish = "stop") =>
	JSON.stringify({
		custom_id: customId,
		response: {
			choices: [{ message: { content }, finish_reason: finish }],
			usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
		},
	});

/** The full happy-path route table: create, upload, job create, poll, download. */
const happyPath = (outputJsonl: string, state = "JOB_STATE_COMPLETED") => [
	{ match: /\/datasets$/, json: {} },
	{ match: /:upload$/, json: {} },
	{ match: /batchInferenceJobs\?/, json: { name: "accounts/acct-42/batchInferenceJobs/job-1" } },
	{
		match: /batchInferenceJobs\/job-1$/,
		json: { state, outputDatasetId: "accounts/acct-42/datasets/out-1" },
	},
	{ match: /:getDownloadEndpoint$/, json: { filenameToSignedUrls: { "r.jsonl": "https://signed.example/r.jsonl" } } },
	{ match: /signed\.example/, text: outputJsonl },
];

beforeEach(() => {
	calls = [];
});
afterEach(() => {
	vi.unstubAllGlobals();
});

describe("fireworks batch — request shape", () => {
	it("emits {custom_id, body} JSONL with NO model, method or url per line", async () => {
		vi.stubGlobal("fetch", mockFetch(happyPath("")));
		await fireworksBatch.submitBatch(model(), [item("a"), item("b")], { env });

		const upload = calls.find((c) => c.url.endsWith(":upload"));
		expect(upload).toBeDefined();
		const form = upload?.init.body as FormData;
		const jsonl = await (form.get("file") as Blob).text();
		const lines = jsonl.split("\n").filter(Boolean);

		expect(lines).toHaveLength(2);
		const first = JSON.parse(lines[0]);
		expect(first).toEqual({ custom_id: "a", body: { messages: [{ role: "user", content: "hello a" }] } });
		// The four OpenAI-isms that must NOT appear.
		expect(first).not.toHaveProperty("method");
		expect(first).not.toHaveProperty("url");
		expect(first.body).not.toHaveProperty("model");
		expect(first).not.toHaveProperty("model");
	});

	it("sets the model once on the JOB, in camelCase, not per line", async () => {
		vi.stubGlobal("fetch", mockFetch(happyPath("")));
		await fireworksBatch.submitBatch(model("accounts/fireworks/models/kimi-k2p7-code"), [item("a")], { env });

		const create = calls.find((c) => c.url.includes("batchInferenceJobs?"));
		const body = JSON.parse(create?.init.body as string);
		expect(body.model).toBe("accounts/fireworks/models/kimi-k2p7-code");
		expect(body.inputDatasetId).toMatch(/^accounts\/acct-42\/datasets\//);
		expect(body.outputDatasetId).toMatch(/^accounts\/acct-42\/datasets\//);
		// camelCase, not snake_case.
		expect(body).not.toHaveProperty("input_dataset_id");
	});

	it("targets the control plane, stripping the inference-plane suffix", async () => {
		vi.stubGlobal("fetch", mockFetch(happyPath("")));
		await fireworksBatch.submitBatch(model(), [item("a")], { env });
		for (const call of calls) {
			expect(call.url).toMatch(/^https:\/\/api\.fireworks\.ai\/v1\/accounts\/acct-42\//);
			expect(call.url).not.toContain("/inference/");
		}
	});

	it("carries systemPrompt, maxTokens and outputSchema into the body", async () => {
		vi.stubGlobal("fetch", mockFetch(happyPath("")));
		const schema = { type: "object", properties: { x: { type: "string" } } };
		await fireworksBatch.submitBatch(
			model(),
			[
				item("a", {
					context: {
						systemPrompt: "be terse",
						messages: [{ role: "user", content: "hi", timestamp: 0 }],
					},
					maxTokens: 256,
					outputSchema: schema,
				}),
			],
			{ env },
		);

		const form = calls.find((c) => c.url.endsWith(":upload"))?.init.body as FormData;
		const line = JSON.parse(await (form.get("file") as Blob).text());
		expect(line.body.messages[0]).toEqual({ role: "system", content: "be terse" });
		expect(line.body.max_tokens).toBe(256);
		expect(line.body.response_format).toEqual({ type: "json_schema", json_schema: { schema } });
	});
});

describe("fireworks batch — configuration faults are loud", () => {
	it("refuses without an account id, and says which variable", async () => {
		vi.stubGlobal("fetch", mockFetch(happyPath("")));
		await expect(
			fireworksBatch.submitBatch(model(), [item("a")], { env: { FIREWORKS_API_KEY: "k" } }),
		).rejects.toThrow(/FIREWORKS_ACCOUNT_ID/);
		expect(calls).toHaveLength(0);
	});

	it("does not mistake the model's publisher account for the caller's", async () => {
		vi.stubGlobal("fetch", mockFetch(happyPath("")));
		await expect(
			fireworksBatch.submitBatch(model(), [item("a")], { env: { FIREWORKS_API_KEY: "k" } }),
		).rejects.toThrow(/that is the model's publisher/);
	});

	it("rejects duplicate and invalid customIds before any network call", async () => {
		vi.stubGlobal("fetch", mockFetch(happyPath("")));
		await expect(fireworksBatch.submitBatch(model(), [item("a"), item("a")], { env })).rejects.toThrow(/a/);
		expect(calls).toHaveLength(0);

		await expect(fireworksBatch.submitBatch(model(), [item("bad id!")], { env })).rejects.toThrow(/invalid customId/);
		expect(calls).toHaveLength(0);
	});

	it("rejects rich Context rather than silently flattening it", async () => {
		vi.stubGlobal("fetch", mockFetch(happyPath("")));
		const withTools: BatchItem = {
			customId: "a",
			context: {
				messages: [{ role: "user", content: "hi", timestamp: 0 }],
				tools: [{ name: "t", description: "d", parameters: {} }],
			} as never,
		};
		await expect(fireworksBatch.submitBatch(model(), [withTools], { env })).rejects.toThrow(/tools/);
		expect(calls).toHaveLength(0);
	});

	it("honours a pre-aborted signal without creating a chargeable job", async () => {
		vi.stubGlobal("fetch", mockFetch(happyPath("")));
		const controller = new AbortController();
		controller.abort();
		await expect(
			fireworksBatch.submitBatch(model(), [item("a")], { env, signal: controller.signal }),
		).rejects.toThrow();
		expect(calls).toHaveLength(0);
	});
});

describe("fireworks batch — results", () => {
	it("maps output lines back by custom_id, with usage", async () => {
		const jsonl = [OUTPUT_LINE("b", "second"), OUTPUT_LINE("a", "first")].join("\n");
		vi.stubGlobal("fetch", mockFetch(happyPath(jsonl)));

		const results = await fireworksBatch.submitAndAwait(model(), [item("a"), item("b")], {
			env,
			pollIntervalMs: 0,
		});

		// Order follows the request, not the response file.
		expect(results.map((r) => r.customId)).toEqual(["a", "b"]);
		expect(results[0]).toMatchObject({ ok: true, text: "first" });
		expect(results[1]).toMatchObject({ ok: true, text: "second" });
		expect(results[0].usage?.totalTokens).toBe(15);
	});

	it("parses and validates outputSchema, failing the item that violates it", async () => {
		const schema = { type: "object", required: ["x"], properties: { x: { type: "string" } } };
		const jsonl = [OUTPUT_LINE("a", '{"x":"ok"}'), OUTPUT_LINE("b", '{"x":123}')].join("\n");
		vi.stubGlobal("fetch", mockFetch(happyPath(jsonl)));

		const results = await fireworksBatch.submitAndAwait(
			model(),
			[item("a", { outputSchema: schema }), item("b", { outputSchema: schema })],
			{ env, pollIntervalMs: 0 },
		);

		expect(results[0]).toMatchObject({ ok: true, value: { x: "ok" } });
		expect(results[1]).toMatchObject({ ok: false, errorKind: "parse" });
	});

	it("reports a token-ceiling stop as truncated, not as a parse failure", async () => {
		const jsonl = OUTPUT_LINE("a", '{"x":"unfinis', "length");
		vi.stubGlobal("fetch", mockFetch(happyPath(jsonl)));

		const results = await fireworksBatch.submitAndAwait(model(), [item("a", { outputSchema: { type: "object" } })], {
			env,
			pollIntervalMs: 0,
		});
		expect(results[0]).toMatchObject({ ok: false, errorKind: "truncated" });
	});

	it("surfaces a per-item provider error without failing the whole job", async () => {
		const jsonl = [
			JSON.stringify({ custom_id: "a", error: { message: "context too long" } }),
			OUTPUT_LINE("b", "fine"),
		].join("\n");
		vi.stubGlobal("fetch", mockFetch(happyPath(jsonl)));

		const results = await fireworksBatch.submitAndAwait(model(), [item("a"), item("b")], {
			env,
			pollIntervalMs: 0,
		});
		expect(results[0]).toMatchObject({ ok: false, errorKind: "provider_item" });
		expect(results[1]).toMatchObject({ ok: true, text: "fine" });
	});

	it("an item absent from the output file still gets a result row", async () => {
		vi.stubGlobal("fetch", mockFetch(happyPath(OUTPUT_LINE("a", "only a"))));
		const results = await fireworksBatch.submitAndAwait(model(), [item("a"), item("missing")], {
			env,
			pollIntervalMs: 0,
		});
		expect(results).toHaveLength(2);
		expect(results[1]).toMatchObject({ customId: "missing", ok: false });
	});

	it("a failed job fails every item with the terminal state named", async () => {
		vi.stubGlobal("fetch", mockFetch(happyPath("", "JOB_STATE_FAILED")));
		const results = await fireworksBatch.submitAndAwait(model(), [item("a")], { env, pollIntervalMs: 0 });
		expect(results[0]).toMatchObject({ ok: false });
		expect(results[0].error).toMatch(/JOB_STATE_FAILED/);
	});

	it("reports the submitted job id through onSubmitted before polling", async () => {
		vi.stubGlobal("fetch", mockFetch(happyPath(OUTPUT_LINE("a", "x"))));
		const seen: string[] = [];
		await fireworksBatch.submitAndAwait(
			model(),
			[item("a")],
			{ env, pollIntervalMs: 0 },
			{
				onSubmitted: (id) => {
					seen.push(id);
				},
			},
		);
		// A real provider-side id, never a synthesised one.
		expect(seen).toEqual(["accounts/acct-42/batchInferenceJobs/job-1"]);
	});

	it("does not send the API key to the signed object-store URL", async () => {
		vi.stubGlobal("fetch", mockFetch(happyPath(OUTPUT_LINE("a", "x"))));
		await fireworksBatch.submitAndAwait(model(), [item("a")], { env, pollIntervalMs: 0 });
		const download = calls.find((c) => c.url.includes("signed.example"));
		const headers = (download?.init.headers ?? {}) as Record<string, string>;
		expect(headers.Authorization).toBeUndefined();
	});
});

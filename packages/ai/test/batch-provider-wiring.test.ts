import { describe, expect, it } from "vitest";
import { createProvider } from "../src/models.ts";
import { type BatchItem, type Model, NotBatchableError, type ProviderBatch } from "../src/types.ts";

const auth = {
	apiKey: { name: "k", login: async () => ({ type: "api_key" as const, key: "k" }), resolve: async () => undefined },
};

const model = (api: string, id = "m"): Model<never> =>
	({
		id,
		name: id,
		api,
		provider: "p",
		baseUrl: "https://example.invalid",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 100,
		maxTokens: 10,
	}) as unknown as Model<never>;

const items: BatchItem[] = [{ customId: "a", context: { messages: [{ role: "user", content: "hi", timestamp: 0 }] } }];

const stubBatch = (tag: string): ProviderBatch => ({
	submitBatch: async () => ({ batchId: `${tag}-1` }),
	pollBatch: async () => [{ customId: "a", ok: true, text: tag }],
	submitAndAwait: async () => [{ customId: "a", ok: true, text: tag }],
});

const streams = { stream: () => ({}) as never, streamSimple: () => ({}) as never };

describe("provider batch wiring", () => {
	it("a provider with no batch transport reports canBatch=false and REJECTS loudly", async () => {
		const p = createProvider({ id: "p", auth, models: [], api: streams });
		expect(p.canBatch(model("anthropic-messages"))).toBe(false);
		await expect(p.submitBatch(model("anthropic-messages"), items)).rejects.toThrow(NotBatchableError);
		await expect(p.submitAndAwaitBatch(model("anthropic-messages"), items)).rejects.toThrow(/NOT_BATCHABLE/);
	});

	it("the rejection names the provider and model and refuses to emulate", async () => {
		const p = createProvider({ id: "vertexish", auth, models: [], api: streams });
		await expect(p.submitBatch(model("google-vertex", "gemini-x"), items)).rejects.toThrow(
			/provider "vertexish".*model "gemini-x"/s,
		);
		// The whole point: no silent realtime fallback.
		await expect(p.submitBatch(model("google-vertex", "gemini-x"), items)).rejects.toThrow(/cannot be emulated/);
	});

	it("a single batch implementation serves every api", async () => {
		const p = createProvider({ id: "p", auth, models: [], api: streams, batch: stubBatch("single") });
		expect(p.canBatch(model("anthropic-messages"))).toBe(true);
		expect(p.canBatch(model("openai-completions"))).toBe(true);
		await expect(p.submitBatch(model("anthropic-messages"), items)).resolves.toEqual({ batchId: "single-1" });
	});

	it("a keyed map dispatches per model.api, and unmapped apis still fail loudly", async () => {
		// Mirrors Fireworks, which serves anthropic-messages AND openai-completions.
		const p = createProvider({
			id: "fireworksish",
			auth,
			models: [],
			api: streams,
			batch: {
				"anthropic-messages": stubBatch("anthropic"),
				"openai-completions": stubBatch("openai"),
			} as never,
		});
		expect(p.canBatch(model("anthropic-messages"))).toBe(true);
		expect(p.canBatch(model("openai-completions"))).toBe(true);
		expect(p.canBatch(model("google-generative-ai"))).toBe(false);

		await expect(p.submitBatch(model("anthropic-messages"), items)).resolves.toEqual({ batchId: "anthropic-1" });
		await expect(p.submitBatch(model("openai-completions"), items)).resolves.toEqual({ batchId: "openai-1" });
		await expect(p.submitBatch(model("google-generative-ai"), items)).rejects.toThrow(NotBatchableError);
	});

	it("pollBatch and submitAndAwaitBatch route through the same capability check", async () => {
		const p = createProvider({ id: "p", auth, models: [], api: streams });
		await expect(p.pollBatch(model("anthropic-messages"), "b1", items)).rejects.toThrow(NotBatchableError);
	});
});

describe("built-in providers declare batch support honestly", () => {
	it("anthropic, openai, google and fireworks can batch; radius cannot", async () => {
		const { anthropicProvider } = await import("../src/providers/anthropic.ts");
		const { googleProvider } = await import("../src/providers/google.ts");
		const { fireworksProvider } = await import("../src/providers/fireworks.ts");
		const { radiusProvider } = await import("../src/providers/radius.ts");

		expect(anthropicProvider().canBatch(model("anthropic-messages"))).toBe(true);
		expect(googleProvider().canBatch(model("google-generative-ai"))).toBe(true);
		// Fireworks serves both wire APIs and batches both.
		expect(fireworksProvider().canBatch(model("anthropic-messages"))).toBe(true);
		expect(fireworksProvider().canBatch(model("openai-completions"))).toBe(true);

		// Radius is a realtime gateway — it must refuse rather than fan out.
		const radius = radiusProvider();
		expect(radius.canBatch(model("pi-messages"))).toBe(false);
		await expect(radius.submitBatch(model("pi-messages"), items)).rejects.toThrow(NotBatchableError);
	});
});

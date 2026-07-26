import { afterEach, describe, expect, it } from "vitest";
import { completeStructured, fauxAssistantMessage, fauxToolCall, registerFauxProvider, Type } from "../src/compat.ts";
import type { Model } from "../src/types.ts";

const registrations: Array<{ unregister: () => void }> = [];

afterEach(() => {
	for (const registration of registrations.splice(0)) {
		registration.unregister();
	}
});

describe("completeStructured", () => {
	const resultSchema = Type.Object({
		answer: Type.String(),
		confidence: Type.Number({ minimum: 0, maximum: 1 }),
	});

	it("uses one internal tool and returns its schema-validated arguments", async () => {
		const registration = registerFauxProvider();
		registrations.push(registration);
		registration.setResponses([
			(context) => {
				expect(context.tools).toEqual([
					expect.objectContaining({
						name: "submit_structured_output",
						parameters: resultSchema,
					}),
				]);
				return fauxAssistantMessage([fauxToolCall("submit_structured_output", { answer: "ok", confidence: 0.9 })], {
					stopReason: "toolUse",
				});
			},
		]);

		const result = await completeStructured(
			registration.getModel(),
			{ messages: [{ role: "user", content: "Return a result", timestamp: Date.now() }] },
			resultSchema,
		);

		expect(result.value).toEqual({ answer: "ok", confidence: 0.9 });
		expect(result.message.stopReason).toBe("toolUse");
	});

	it("accepts plain JSON Schema for Zod-derived provider constraints", async () => {
		const registration = registerFauxProvider();
		registrations.push(registration);
		registration.setResponses([
			fauxAssistantMessage([fauxToolCall("submit_structured_output", { answer: "ok" })], {
				stopReason: "toolUse",
			}),
		]);
		const jsonSchema = {
			type: "object",
			properties: { answer: { type: "string" } },
			required: ["answer"],
			additionalProperties: false,
		};

		const result = await completeStructured(
			registration.getModel(),
			{ messages: [{ role: "user", content: "Return a result", timestamp: Date.now() }] },
			jsonSchema,
		);

		expect(result.value).toEqual({ answer: "ok" });
	});

	it("rejects missing structured tool calls instead of parsing assistant text", async () => {
		const registration = registerFauxProvider();
		registrations.push(registration);
		registration.setResponses([fauxAssistantMessage('{"answer":"not accepted"}')]);

		await expect(
			completeStructured(
				registration.getModel(),
				{ messages: [{ role: "user", content: "Return a result", timestamp: Date.now() }] },
				resultSchema,
			),
		).rejects.toThrow("expected exactly one submit_structured_output call");
	});

	it("rejects tool arguments that fail the original schema", async () => {
		const registration = registerFauxProvider();
		registrations.push(registration);
		registration.setResponses([
			fauxAssistantMessage([fauxToolCall("submit_structured_output", { answer: "ok", confidence: 2 })], {
				stopReason: "toolUse",
			}),
		]);

		await expect(
			completeStructured(
				registration.getModel(),
				{ messages: [{ role: "user", content: "Return a result", timestamp: Date.now() }] },
				resultSchema,
			),
		).rejects.toThrow('Validation failed for tool "submit_structured_output"');
	});

	it("rejects Anthropic-unsupported schema keywords before any request, naming the paths", async () => {
		const model: Model<"anthropic-messages"> = {
			id: "claude-opus-4-8",
			name: "Claude Opus 4.8",
			api: "anthropic-messages",
			provider: "anthropic",
			baseUrl: "https://api.anthropic.com",
			reasoning: false,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 1_000,
			maxTokens: 100,
		};
		await expect(
			completeStructured(
				model,
				{ messages: [{ role: "user", content: "Return", timestamp: Date.now() }] },
				resultSchema,
				{ apiKey: "test-key" },
			),
		).rejects.toThrow(/properties\.confidence\.minimum, properties\.confidence\.maximum/);
	});

	it("sends the native Anthropic json_schema output format for a supported schema", async () => {
		const structuralSchema = Type.Object({ answer: Type.String() });
		const model: Model<"anthropic-messages"> = {
			id: "claude-opus-4-8",
			name: "Claude Opus 4.8",
			api: "anthropic-messages",
			provider: "anthropic",
			baseUrl: "https://api.anthropic.com",
			reasoning: false,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 1_000,
			maxTokens: 100,
		};
		let payload: unknown;
		await expect(
			completeStructured(
				model,
				{ messages: [{ role: "user", content: "Return", timestamp: Date.now() }] },
				structuralSchema,
				{
					apiKey: "test-key",
					onPayload: (nextPayload) => {
						payload = nextPayload;
						throw new Error("captured");
					},
				},
			),
		).rejects.toThrow("captured");
		expect(payload).toMatchObject({
			output_config: { format: { type: "json_schema" } },
		});
	});

	it("uses Bedrock native JSON Schema output instead of an internal tool", async () => {
		const model: Model<"bedrock-converse-stream"> = {
			id: "us.anthropic.claude-sonnet-4-6",
			name: "Bedrock Claude Test",
			api: "bedrock-converse-stream",
			provider: "test-provider",
			baseUrl: "https://bedrock-runtime.us-east-1.amazonaws.com",
			reasoning: false,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 1_000,
			maxTokens: 100,
		};
		let payload: unknown;

		await expect(
			completeStructured(
				model,
				{ messages: [{ role: "user", content: "Return a result", timestamp: Date.now() }] },
				resultSchema,
				{
					apiKey: "test-key",
					onPayload: (nextPayload) => {
						payload = nextPayload;
						throw new Error("stop after payload capture");
					},
				},
			),
		).rejects.toThrow("stop after payload capture");

		expect(payload).toMatchObject({
			outputConfig: {
				textFormat: {
					type: "json_schema",
					structure: {
						jsonSchema: { name: "submit_structured_output" },
					},
				},
			},
		});
		expect(payload).toMatchObject({ toolConfig: undefined });
	});

	it("rejects unsupported Bedrock models rather than silently using a non-constrained fallback", async () => {
		const model: Model<"bedrock-converse-stream"> = {
			id: "us.anthropic.claude-opus-4-8",
			name: "Bedrock Claude Opus 4.8",
			api: "bedrock-converse-stream",
			provider: "test-provider",
			baseUrl: "https://bedrock-runtime.us-east-1.amazonaws.com",
			reasoning: false,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 1_000,
			maxTokens: 100,
		};

		await expect(
			completeStructured(
				model,
				{ messages: [{ role: "user", content: "Return a result", timestamp: Date.now() }] },
				resultSchema,
			),
		).rejects.toThrow("Native Bedrock structured output is unsupported");
	});

	it("forces the output tool in OpenAI Responses payloads", async () => {
		const model: Model<"openai-responses"> = {
			id: "test-model",
			name: "Test model",
			api: "openai-responses",
			provider: "test-provider",
			baseUrl: "https://example.invalid/v1",
			reasoning: false,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 1_000,
			maxTokens: 100,
		};
		let payload: unknown;

		await expect(
			completeStructured(
				model,
				{ messages: [{ role: "user", content: "Return a result", timestamp: Date.now() }] },
				resultSchema,
				{
					apiKey: "test-key",
					onPayload: (nextPayload) => {
						payload = nextPayload;
						throw new Error("stop after payload capture");
					},
				},
			),
		).rejects.toThrow("stop after payload capture");

		// 0.82.1 merge: `strict` is now PROVIDER-GATED, not forced by this helper.
		// Upstream's openai-responses compat defaults `supportsStrictMode` to false
		// ("Defaults are API-specific; generated OpenAI models enable it
		// explicitly" — types.ts), so a synthetic model with no compat gets an
		// unstrict tool. Real catalog models (openai, azure) enable it, so
		// production behaviour is unchanged. The output tool is still FORCED via
		// tool_choice regardless — that is what this test exists to pin.
		expect(payload).toMatchObject({
			tool_choice: { type: "function", name: "submit_structured_output" },
			tools: [
				{
					type: "function",
					name: "submit_structured_output",
				},
			],
		});
		expect((payload as { tools: { strict?: unknown }[] }).tools[0].strict).toBeUndefined();
	});

	it("applies strict to the output tool when the provider declares support", async () => {
		const model: Model<"openai-responses"> = {
			id: "test-model",
			name: "Test model",
			api: "openai-responses",
			provider: "test-provider",
			baseUrl: "https://example.invalid/v1",
			reasoning: false,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 1_000,
			maxTokens: 100,
			compat: { supportsStrictMode: true },
		};
		let payload: unknown;

		await expect(
			completeStructured(
				model,
				{ messages: [{ role: "user", content: "Return a result", timestamp: Date.now() }] },
				resultSchema,
				{
					apiKey: "test-key",
					onPayload: (nextPayload) => {
						payload = nextPayload;
						throw new Error("stop after payload capture");
					},
				},
			),
		).rejects.toThrow("stop after payload capture");

		// completeStructured requests `constrainedSampling: { type: "json_schema",
		// strict: "prefer" }` on its internal output tool (compat.ts). "prefer"
		// rather than "require" because this is a fallback path with a
		// validate-and-retry backstop — losing provider strictness must degrade,
		// never throw.
		expect(payload).toMatchObject({
			tools: [
				{
					type: "function",
					name: "submit_structured_output",
					strict: true,
				},
			],
		});
	});
});

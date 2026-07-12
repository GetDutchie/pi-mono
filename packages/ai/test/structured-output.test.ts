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

		expect(payload).toMatchObject({
			tool_choice: { type: "function", name: "submit_structured_output" },
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

/**
 * Focused coverage for opt-in per-tool `strict` propagation across the
 * providers that natively strictify tool schemas: OpenAI Chat Completions,
 * OpenAI Responses (shared with Azure), and Anthropic Messages.
 *
 * Ordinary tools (no `strict` field) must never be strictified. Only tools
 * with `strict: true` should get a provider-native strict schema. The
 * PI_STRICT_TOOLS=0 environment variable remains a global emergency kill
 * switch that forces every tool - including opted-in ones - non-strict.
 */

import { Type } from "typebox";
import { describe, expect, it } from "vitest";
import { stream as streamAnthropic } from "../src/api/anthropic-messages.ts";
import { stream as streamAzureOpenAIResponses } from "../src/api/azure-openai-responses.ts";
import { stream as streamOpenAICompletions } from "../src/api/openai-completions.ts";
import { stream as streamOpenAIResponses } from "../src/api/openai-responses.ts";
import { getModel } from "../src/compat.ts";
import type { Context, Model, StreamOptions, Tool } from "../src/types.ts";

class PayloadCaptured extends Error {}

function ordinaryTool(name: string): Tool {
	return {
		name,
		description: `The ${name} tool`,
		parameters: Type.Object({ value: Type.Optional(Type.String()) }),
	};
}

function strictTool(name: string): Tool {
	return {
		name,
		description: `The ${name} tool`,
		parameters: Type.Object({ value: Type.Optional(Type.String()) }),
		strict: true,
	};
}

function unstrictifiableTool(name: string): Tool {
	return {
		name,
		description: `The ${name} tool`,
		parameters: Type.Object({ values: Type.Record(Type.String(), Type.String()) }),
		strict: true,
	};
}

function makeContext(tools: Tool[]): Context {
	return {
		messages: [{ role: "user", content: "Call a tool", timestamp: Date.now() }],
		tools,
	};
}

async function capturePayload<T>(
	streamFn: (
		model: Model<any>,
		context: Context,
		options: StreamOptions & Record<string, unknown>,
	) => { result: () => Promise<unknown> },
	model: Model<any>,
	context: Context,
	extraOptions: Record<string, unknown> = {},
): Promise<T> {
	let captured: T | undefined;
	const s = streamFn(model, { ...context, messages: [...context.messages] } as Context, {
		apiKey: "fake-key",
		...extraOptions,
		onPayload: (payload: unknown) => {
			captured = payload as T;
			throw new PayloadCaptured();
		},
	});
	await s.result();
	if (!captured) throw new Error("Expected payload capture");
	return captured;
}

interface OpenAIChatToolPayload {
	tools?: Array<{ function?: { name: string; strict?: boolean; parameters?: Record<string, unknown> } }>;
}

interface OpenAIResponsesToolPayload {
	tools?: Array<{ name: string; strict?: boolean | null; parameters?: Record<string, unknown> }>;
}

interface AnthropicToolPayload {
	tools?: Array<{ name: string; strict?: boolean; input_schema?: Record<string, unknown> }>;
}

function chatOpenAIModel(): Model<"openai-completions"> {
	return {
		...getModel("openai", "gpt-4o-mini"),
		api: "openai-completions",
		baseUrl: "http://127.0.0.1:9",
	} as Model<"openai-completions">;
}

function responsesOpenAIModel(): Model<"openai-responses"> {
	return { ...getModel("openai", "gpt-5.4"), baseUrl: "http://127.0.0.1:9" } as Model<"openai-responses">;
}

function anthropicModel(): Model<"anthropic-messages"> {
	return { ...getModel("anthropic", "claude-opus-4-6"), baseUrl: "http://127.0.0.1:9" } as Model<"anthropic-messages">;
}

describe("per-tool strict opt-in: OpenAI Chat Completions", () => {
	it("does not strictify an ordinary tool", async () => {
		const payload = await capturePayload<OpenAIChatToolPayload>(
			streamOpenAICompletions,
			chatOpenAIModel(),
			makeContext([ordinaryTool("read_file")]),
		);
		expect(payload.tools?.[0]?.function?.strict).toBe(false);
	});

	it("strictifies only an opted-in tool in a mixed batch", async () => {
		const payload = await capturePayload<OpenAIChatToolPayload>(
			streamOpenAICompletions,
			chatOpenAIModel(),
			makeContext([ordinaryTool("read_file"), strictTool("structured_output")]),
		);
		const ordinary = payload.tools?.find((t) => t.function?.name === "read_file");
		const strict = payload.tools?.find((t) => t.function?.name === "structured_output");
		expect(ordinary?.function?.strict).toBe(false);
		expect(ordinary?.function?.parameters).toMatchObject({
			properties: { value: { type: "string" } },
		});
		expect(strict?.function?.strict).toBe(true);
	});

	it("PI_STRICT_TOOLS=0 kill switch forces an opted-in tool non-strict", async () => {
		const payload = await capturePayload<OpenAIChatToolPayload>(
			streamOpenAICompletions,
			chatOpenAIModel(),
			makeContext([strictTool("structured_output")]),
			{ env: { PI_STRICT_TOOLS: "0" } },
		);
		expect(payload.tools?.[0]?.function?.strict).toBe(false);
	});

	it("falls back to the original schema when an opted-in tool is unstrictifiable", async () => {
		const payload = await capturePayload<OpenAIChatToolPayload>(
			streamOpenAICompletions,
			chatOpenAIModel(),
			makeContext([unstrictifiableTool("map_output")]),
		);
		expect(payload.tools?.[0]?.function?.strict).toBe(false);
		expect(payload.tools?.[0]?.function?.parameters).toHaveProperty("properties.values.patternProperties");
	});
});

describe("per-tool strict opt-in: OpenAI Responses", () => {
	it("does not strictify an ordinary tool", async () => {
		const payload = await capturePayload<OpenAIResponsesToolPayload>(
			streamOpenAIResponses,
			responsesOpenAIModel(),
			makeContext([ordinaryTool("read_file")]),
		);
		expect(payload.tools?.[0]?.strict).toBe(false);
	});

	it("strictifies only an opted-in tool in a mixed batch", async () => {
		const payload = await capturePayload<OpenAIResponsesToolPayload>(
			streamOpenAIResponses,
			responsesOpenAIModel(),
			makeContext([ordinaryTool("read_file"), strictTool("structured_output")]),
		);
		const ordinary = payload.tools?.find((t) => t.name === "read_file");
		const strict = payload.tools?.find((t) => t.name === "structured_output");
		expect(ordinary?.strict).toBe(false);
		expect(ordinary?.parameters).toMatchObject({ properties: { value: { type: "string" } } });
		expect(strict?.strict).toBe(true);
	});

	it("PI_STRICT_TOOLS=0 kill switch forces an opted-in tool non-strict", async () => {
		const payload = await capturePayload<OpenAIResponsesToolPayload>(
			streamOpenAIResponses,
			responsesOpenAIModel(),
			makeContext([strictTool("structured_output")]),
			{ env: { PI_STRICT_TOOLS: "0" } },
		);
		expect(payload.tools?.[0]?.strict).toBe(false);
	});

	it("falls back to the original schema when an opted-in tool is unstrictifiable", async () => {
		const payload = await capturePayload<OpenAIResponsesToolPayload>(
			streamOpenAIResponses,
			responsesOpenAIModel(),
			makeContext([unstrictifiableTool("map_output")]),
		);
		expect(payload.tools?.[0]?.strict).toBe(false);
		expect(payload.tools?.[0]?.parameters).toHaveProperty("properties.values.patternProperties");
	});
});

describe("per-tool strict opt-in: Azure OpenAI Responses", () => {
	it("does not strictify an ordinary tool", async () => {
		const model = getModel("azure-openai-responses", "gpt-4o-mini");
		const payload = await capturePayload<OpenAIResponsesToolPayload>(
			streamAzureOpenAIResponses,
			model,
			makeContext([ordinaryTool("read_file")]),
			{ azureBaseUrl: "http://127.0.0.1:9" },
		);
		expect(payload.tools?.[0]?.strict).toBe(false);
	});

	it("strictifies only an opted-in tool in a mixed batch", async () => {
		const model = getModel("azure-openai-responses", "gpt-4o-mini");
		const payload = await capturePayload<OpenAIResponsesToolPayload>(
			streamAzureOpenAIResponses,
			model,
			makeContext([ordinaryTool("read_file"), strictTool("structured_output")]),
			{ azureBaseUrl: "http://127.0.0.1:9" },
		);
		const ordinary = payload.tools?.find((t) => t.name === "read_file");
		const strict = payload.tools?.find((t) => t.name === "structured_output");
		expect(ordinary?.strict).toBe(false);
		expect(strict?.strict).toBe(true);
	});
});

describe("per-tool strict opt-in: Anthropic Messages", () => {
	it("does not strictify an ordinary tool", async () => {
		const payload = await capturePayload<AnthropicToolPayload>(
			streamAnthropic,
			anthropicModel(),
			makeContext([ordinaryTool("read_file")]),
		);
		expect(payload.tools?.[0]?.strict).toBeUndefined();
	});

	it("strictifies only an opted-in tool in a mixed batch", async () => {
		const payload = await capturePayload<AnthropicToolPayload>(
			streamAnthropic,
			anthropicModel(),
			makeContext([ordinaryTool("read_file"), strictTool("structured_output")]),
		);
		const ordinary = payload.tools?.find((t) => t.name === "read_file");
		const strict = payload.tools?.find((t) => t.name === "structured_output");
		expect(ordinary?.strict).toBeUndefined();
		expect(ordinary?.input_schema).toMatchObject({ properties: { value: { type: "string" } } });
		expect(strict?.strict).toBe(true);
	});

	it("PI_STRICT_TOOLS=0 kill switch forces an opted-in tool non-strict", async () => {
		const payload = await capturePayload<AnthropicToolPayload>(
			streamAnthropic,
			anthropicModel(),
			makeContext([strictTool("structured_output")]),
			{ env: { PI_STRICT_TOOLS: "0" } },
		);
		expect(payload.tools?.[0]?.strict).toBeUndefined();
	});

	it("falls back to the original schema when an opted-in tool is unstrictifiable", async () => {
		const payload = await capturePayload<AnthropicToolPayload>(
			streamAnthropic,
			anthropicModel(),
			makeContext([unstrictifiableTool("map_output")]),
		);
		expect(payload.tools?.[0]?.strict).toBeUndefined();
		expect(payload.tools?.[0]?.input_schema).toHaveProperty("properties.values.patternProperties");
	});
});

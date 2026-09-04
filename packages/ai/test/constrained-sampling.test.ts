import type { ResponseStreamEvent } from "openai/resources/responses/responses.js";
import { Type } from "typebox";
import { describe, expect, it } from "vitest";
import {
	appendGrammarToolInputJsonDelta,
	getJsonSchemaToolParameters,
	makeStrictJsonSchema,
	resolveJsonSchemaStrictSampling,
} from "../src/api/constrained-sampling.ts";
import {
	convertResponsesMessages,
	convertResponsesTools,
	processResponsesStream,
} from "../src/api/openai-responses-shared.ts";
import type { AssistantMessage, Context, Model, Tool, ToolCall } from "../src/types.ts";
import { AssistantMessageEventStream } from "../src/utils/event-stream.ts";
import { anthropicStrictToolSchema, strictToolSchema } from "../src/utils/strict-tool-schema.ts";

function makeModel(): Model<"openai-responses"> {
	return {
		id: "gpt-test",
		name: "GPT Test",
		api: "openai-responses",
		provider: "openai",
		baseUrl: "https://api.openai.com/v1",
		reasoning: false,
		input: ["text", "image"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128000,
		maxTokens: 4096,
	};
}

function makeUsage(): AssistantMessage["usage"] {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function makeOutput(): AssistantMessage {
	return {
		role: "assistant",
		content: [],
		api: "openai-responses",
		provider: "openai",
		model: "gpt-test",
		usage: makeUsage(),
		stopReason: "pending",
		timestamp: Date.now(),
	};
}

async function* iterateEvents(events: ResponseStreamEvent[]): AsyncGenerator<ResponseStreamEvent> {
	yield* events;
}

function makeTool(overrides: Partial<Tool> = {}): Tool {
	return {
		name: "sample_tool",
		description: "Sample tool",
		parameters: Type.Object({ payload: Type.String() }, { additionalProperties: false }),
		...overrides,
	};
}

function captureToolCallEvents(stream: AssistantMessageEventStream): {
	starts: ToolCall["arguments"][];
	deltas: string[];
} {
	const starts: ToolCall["arguments"][] = [];
	const deltas: string[] = [];
	const originalPush = stream.push.bind(stream);
	stream.push = (event) => {
		if (event.type === "toolcall_start") {
			const block = event.partial.content[event.contentIndex];
			if (block?.type === "toolCall") starts.push(structuredClone(block.arguments));
		} else if (event.type === "toolcall_delta") {
			deltas.push(event.delta);
		}
		originalPush(event);
	};
	return { starts, deltas };
}

describe("constrained tool sampling", () => {
	it("converts supported constraints and falls back when unsupported", () => {
		expect(
			convertResponsesTools([makeTool({ constrainedSampling: { type: "json_schema", strict: "prefer" } })])[0],
		).toMatchObject({ type: "function", name: "sample_tool", strict: true });

		expect(() =>
			convertResponsesTools([makeTool({ constrainedSampling: { type: "json_schema", strict: "require" } })], {
				supportsStrictMode: false,
			}),
		).toThrow('Tool "sample_tool" requires JSON-schema constrained sampling');

		const grammarTool = makeTool({
			constrainedSampling: { type: "grammar", variants: { openai_lark: "start: /[a-z]+/" } },
		});
		expect(convertResponsesTools([grammarTool], { supportsOpenAIGrammarTools: true })[0]).toMatchObject({
			type: "custom",
			name: "sample_tool",
			format: { type: "grammar", syntax: "lark", definition: "start: /[a-z]+/" },
		});
		expect(() =>
			convertResponsesTools([makeTool({ constrainedSampling: { type: "grammar", variants: {} } })], {
				supportsOpenAIGrammarTools: true,
			}),
		).toThrow(
			'Tool "sample_tool" cannot use grammar constrained sampling: no supported grammar variant was provided',
		);

		const fallback = convertResponsesTools([grammarTool], {
			supportsOpenAIGrammarTools: false,
			supportsStrictMode: false,
		})[0];
		expect(fallback).toMatchObject({ type: "function", name: "sample_tool" });
		expect("strict" in (fallback as object)).toBe(false);

		expect(convertResponsesTools([makeTool({ constrainedSampling: false })])).toEqual(
			convertResponsesTools([makeTool()]),
		);
	});

	it("derives strict provider schemas without changing tool definitions", () => {
		const parameters = Type.Object({
			path: Type.String(),
			offset: Type.Optional(Type.Number()),
			metadata: Type.Object({ enabled: Type.Optional(Type.Boolean()) }),
			nullable: Type.Optional(Type.Union([Type.String(), Type.Null()])),
		});

		const strict = makeStrictJsonSchema(parameters);

		expect(parameters).not.toHaveProperty("additionalProperties");
		expect(parameters.required).toEqual(["path", "metadata"]);
		expect(strict).toMatchObject({
			additionalProperties: false,
			required: ["path", "offset", "metadata", "nullable"],
			properties: {
				// A plainly-typed optional becomes a nullable type union, which is the
				// form OpenAI's strict-mode documentation uses.
				offset: { type: ["number", "null"] },
				metadata: {
					additionalProperties: false,
					required: ["enabled"],
					properties: { enabled: { type: ["boolean", "null"] } },
				},
				// An optional that already admits null is left alone.
				nullable: { anyOf: [{ type: "string" }, { type: "null" }] },
			},
		});
	});

	it("falls back or rejects schemas that cannot be safely converted", () => {
		const cases: Array<{ parameters: Tool["parameters"]; error: string }> = [
			{
				parameters: Type.Object({ metadata: Type.Object({}, { additionalProperties: Type.String() }) }),
				error: "additionalProperties is unsupported",
			},
			{
				// An Intersect root is an allOf composition with no type of its own, and
				// tool parameters must have an object root in every dialect.
				parameters: Type.Intersect([Type.Object({ a: Type.String() }), Type.Object({ b: Type.Number() })]),
				error: "root schema must have type object",
			},
			{
				parameters: Type.Object({ merged: Type.Intersect([Type.Object({ a: Type.String() })]) }),
				error: "allOf is unsupported",
			},
			{
				parameters: Type.Object({ pairs: Type.Tuple([Type.String(), Type.Number()]) }),
				error: "tuple items is unsupported",
			},
			{
				parameters: Type.Object({
					guarded: { not: { type: "string" } } as unknown as ReturnType<typeof Type.String>,
				}),
				error: "not is unsupported",
			},
		];

		for (const { parameters, error } of cases) {
			const tool: Tool = {
				...makeTool(),
				parameters,
				constrainedSampling: { type: "json_schema", strict: "prefer" },
			};

			expect(() => makeStrictJsonSchema(parameters)).toThrow(error);
			expect(resolveJsonSchemaStrictSampling(tool, true)).toBeUndefined();
			expect(convertResponsesTools([tool], { supportsStrictMode: true })[0]).toMatchObject({
				strict: false,
				parameters,
			});

			tool.constrainedSampling = { type: "json_schema", strict: "require" };
			expect(() => resolveJsonSchemaStrictSampling(tool, true)).toThrow(error);
		}
	});

	it("strips validation keywords a strict request would be rejected for", () => {
		// OpenAI strict mode rejects these outright, so leaving them in the wire
		// schema turns constrained sampling into a hard request failure. They stay
		// enforced post-hoc by original-schema validation.
		const parameters = Type.Object({
			name: Type.String({ minLength: 1, maxLength: 8, pattern: "^[a-z]+$", format: "email" }),
			count: Type.Number({ minimum: 0, maximum: 10, multipleOf: 2 }),
			tags: Type.Array(Type.String(), { minItems: 1, maxItems: 3, uniqueItems: true }),
		});

		const strict = JSON.stringify(makeStrictJsonSchema(parameters));

		for (const keyword of [
			"minLength",
			"maxLength",
			"pattern",
			"format",
			"minimum",
			"maximum",
			"multipleOf",
			"minItems",
			"maxItems",
			"uniqueItems",
		]) {
			expect(strict).not.toContain(keyword);
		}
		// The original tool definition is untouched.
		expect(JSON.stringify(parameters)).toContain("minLength");
	});

	it("keeps reused sub-schemas strict via $defs instead of falling back", () => {
		// TypeBox emits $ref/$defs for any reused sub-schema, so refusing them
		// would drop such tools out of constrained sampling entirely.
		const Inner = Type.Object({ a: Type.String(), b: Type.Optional(Type.Number()) }, { $id: "Inner" });
		const parameters = {
			type: "object",
			properties: { first: { $ref: "#/$defs/Inner" }, second: { $ref: "#/$defs/Inner" } },
			required: ["first"],
			$defs: { Inner },
		} as unknown as Tool["parameters"];

		const strict = makeStrictJsonSchema(parameters) as Record<string, any>;

		expect(strict.required).toEqual(["first", "second"]);
		// An optional $ref property gets a null escape so the model can say "absent".
		expect(strict.properties.second).toEqual({ anyOf: [{ $ref: "#/$defs/Inner" }, { type: "null" }] });
		// The definition pool is a map of schemas, and each entry must itself be strict.
		expect(strict.$defs.Inner).toMatchObject({
			additionalProperties: false,
			required: ["a", "b"],
			properties: { b: { type: ["number", "null"] } },
		});
	});

	it("rewrites oneOf to anyOf and keeps object unions strict", () => {
		const parameters = {
			type: "object",
			properties: {
				choice: { oneOf: [{ type: "string" }, { type: "number" }] },
				shape: { anyOf: [{ type: "object", properties: { nested: { type: "string" } } }, { type: "null" }] },
			},
			required: ["choice", "shape"],
		} as unknown as Tool["parameters"];

		const strict = makeStrictJsonSchema(parameters) as Record<string, any>;

		// Strict mode understands anyOf but not oneOf.
		expect(strict.properties.choice).toEqual({ anyOf: [{ type: "string" }, { type: "number" }] });
		expect(JSON.stringify(strict)).not.toContain("oneOf");
		// Object branches inside a union are strictified rather than rejected.
		expect(strict.properties.shape.anyOf[0]).toMatchObject({ additionalProperties: false, required: ["nested"] });
	});

	it("refuses strict for references a provider grammar compiler cannot resolve", () => {
		// A provider only sees the document we send, so an external ref is
		// unfetchable and a dangling pointer unresolvable. Recursive reference
		// graphs are not accepted across the strict subsets we target, notably
		// Amazon Bedrock, which has 72 strict-capable models. All three must
		// degrade to an unstrict tool rather than 400 the whole request.
		const cases: Array<{ label: string; parameters: Tool["parameters"]; error: string }> = [
			{
				label: "external",
				parameters: {
					type: "object",
					properties: { child: { $ref: "https://example.com/child.json" } },
					required: ["child"],
				} as Tool["parameters"],
				error: "a non-local $ref",
			},
			{
				label: "dangling",
				parameters: {
					type: "object",
					properties: { child: { $ref: "#/$defs/missing" } },
					required: ["child"],
				} as Tool["parameters"],
				error: "an unresolvable $ref",
			},
			{
				label: "recursive",
				parameters: {
					type: "object",
					properties: { node: { $ref: "#/$defs/node" } },
					required: ["node"],
					$defs: { node: { type: "object", properties: { next: { $ref: "#/$defs/node" } }, required: [] } },
				} as Tool["parameters"],
				error: "a recursive $ref",
			},
		];

		for (const { parameters, error } of cases) {
			expect(() => makeStrictJsonSchema(parameters)).toThrow(error);
			expect(() => makeStrictJsonSchema(parameters, "anthropic")).toThrow(error);
			const tool: Tool = {
				...makeTool(),
				parameters,
				constrainedSampling: { type: "json_schema", strict: "prefer" },
			};
			expect(resolveJsonSchemaStrictSampling(tool, true)).toBeUndefined();
			tool.constrainedSampling = { type: "json_schema", strict: "require" };
			expect(() => resolveJsonSchemaStrictSampling(tool, true)).toThrow(error);
		}

		// A shared, local, acyclic reference is the case worth supporting and must
		// still strictify.
		const shared = {
			type: "object",
			properties: { a: { $ref: "#/$defs/I" }, b: { $ref: "#/$defs/I" } },
			required: ["a", "b"],
			$defs: { I: { type: "object", properties: { v: { type: "string" } }, required: ["v"] } },
		} as unknown as Tool["parameters"];
		expect(() => makeStrictJsonSchema(shared)).not.toThrow();
	});

	it("requires an object root for tool parameters but not for Anthropic structured output", () => {
		const rootUnion = {
			anyOf: [
				{ type: "object", properties: { a: { type: "string" } }, required: ["a"] },
				{ type: "object", properties: { b: { type: "number" } }, required: [] },
			],
		} as unknown as Tool["parameters"];

		// Tool parameters: a root union has no `properties` to build a grammar from,
		// and Anthropic's input_schema must be an object, so both dialects degrade.
		for (const dialect of ["openai", "anthropic"] as const) {
			expect(() => makeStrictJsonSchema(rootUnion, dialect)).toThrow("root schema must have type object");
			const tool: Tool = {
				...makeTool(),
				parameters: rootUnion,
				constrainedSampling: { type: "json_schema", strict: "prefer" },
			};
			expect(resolveJsonSchemaStrictSampling(tool, true, dialect)).toBeUndefined();
			expect(getJsonSchemaToolParameters(tool, undefined, dialect)).toBe(rootUnion);
		}

		// Structured output is a different surface: Anthropic's own transformer
		// handles a root union and output_config accepts one, so requiring an object
		// root there would push callers onto their raw-schema fallback and send an
		// UNtransformed union with open object branches.
		const anthropicOutput = anthropicStrictToolSchema(rootUnion as object) as Record<string, any>;
		expect(anthropicOutput).not.toBeNull();
		expect(anthropicOutput.anyOf[0].additionalProperties).toBe(false);
		// The OpenAI-family strict subset does require an object root.
		expect(strictToolSchema(rootUnion as object)).toBeNull();
	});

	it("uses Anthropic's own transformer for the anthropic dialect", () => {
		const parameters = Type.Object({ path: Type.String({ minLength: 1 }), offset: Type.Optional(Type.Number()) });

		const anthropic = makeStrictJsonSchema(parameters, "anthropic") as Record<string, any>;

		expect(anthropic.additionalProperties).toBe(false);
		// Anthropic's transformer preserves unsupported keywords by moving them
		// into the description rather than dropping them silently.
		expect(JSON.stringify(anthropic)).not.toContain('"minLength"');
		// A probe must use the same dialect it will produce with.
		const tool: Tool = {
			...makeTool(),
			parameters,
			constrainedSampling: { type: "json_schema", strict: "require" },
		};
		expect(resolveJsonSchemaStrictSampling(tool, true, "anthropic")).toBe(true);
	});

	it("replays grammar calls as custom Responses items", () => {
		const replayedToolCall: ToolCall = {
			type: "toolCall",
			id: "call_1|ctc_1",
			name: "sample_tool",
			arguments: { payload: "abc" },
		};
		const context: Context = {
			messages: [
				{
					role: "assistant",
					api: "openai-responses",
					provider: "openai",
					model: "gpt-test",
					content: [replayedToolCall],
					usage: makeUsage(),
					stopReason: "toolUse",
					timestamp: Date.now(),
				},
				{
					role: "toolResult",
					toolCallId: "call_1|ctc_1",
					toolName: "sample_tool",
					content: [{ type: "text", text: "done" }],
					isError: false,
					timestamp: Date.now(),
				},
			],
		};
		for (const invalidArguments of [{}, { payload: 42 }]) {
			replayedToolCall.arguments = invalidArguments;
			expect(() =>
				convertResponsesMessages(makeModel(), context, new Set(["openai"]), {
					grammarToolInputProperties: new Map([["sample_tool", "payload"]]),
				}),
			).toThrow('Grammar tool call "sample_tool" requires argument "payload" to be a string');
		}

		replayedToolCall.arguments = { payload: "abc" };
		const messages = convertResponsesMessages(makeModel(), context, new Set(["openai"]), {
			grammarToolInputProperties: new Map([["sample_tool", "payload"]]),
		});

		expect(messages).toContainEqual({
			type: "custom_tool_call",
			id: "ctc_1",
			call_id: "call_1",
			name: "sample_tool",
			input: "abc",
		});
		expect(messages).toContainEqual({
			type: "custom_tool_call_output",
			call_id: "call_1",
			output: "done",
		});
	});

	it("keeps grammar input JSON deltas append-only", () => {
		const buffer = { input: "", started: false, closed: false };
		const first = appendGrammarToolInputJsonDelta(buffer, "payload", 'a"', false);
		const second = appendGrammarToolInputJsonDelta(buffer, "payload", 'a"\nb', true);

		expect(JSON.parse(`${first}${second}`)).toEqual({ payload: 'a"\nb' });
		expect(appendGrammarToolInputJsonDelta(buffer, "payload", 'a"\nb', true)).toBeUndefined();
		expect(() => appendGrammarToolInputJsonDelta(buffer, "payload", "changed", true)).toThrow(
			'grammar tool input for property "payload" changed after it was closed',
		);
	});

	it("starts custom Responses tool calls with their initial input", async () => {
		const output = makeOutput();
		const stream = new AssistantMessageEventStream();
		const { starts, deltas } = captureToolCallEvents(stream);
		const events = [
			{
				type: "response.output_item.added",
				output_index: 0,
				item: { type: "custom_tool_call", call_id: "call_1", id: "ctc_1", name: "sample_tool", input: "a" },
			},
			{
				type: "response.custom_tool_call_input.delta",
				output_index: 0,
				item_id: "ctc_1",
				delta: "b",
			},
			{
				type: "response.custom_tool_call_input.done",
				output_index: 0,
				item_id: "ctc_1",
				input: "abc",
			},
			{
				type: "response.output_item.done",
				output_index: 0,
				item: { type: "custom_tool_call", call_id: "call_1", id: "ctc_1", name: "sample_tool", input: "abc" },
			},
			{
				type: "response.completed",
				response: { status: "completed", usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 } },
			},
		] as ResponseStreamEvent[];

		await processResponsesStream(iterateEvents(events), output, stream, makeModel(), {
			grammarToolInputProperties: new Map([["sample_tool", "payload"]]),
		});

		expect(output.stopReason).toBe("toolUse");
		expect(starts).toEqual([{ payload: "a" }]);
		expect(output.content).toEqual([
			{ type: "toolCall", id: "call_1|ctc_1", name: "sample_tool", arguments: { payload: "abc" } },
		]);
		expect(JSON.parse(deltas.join(""))).toEqual({ payload: "abc" });
	});
});

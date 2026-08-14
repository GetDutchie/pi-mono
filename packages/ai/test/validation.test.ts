import { Type } from "typebox";
import { Compile } from "typebox/compile";
import { describe, expect, it } from "vitest";
import type { Tool, ToolCall } from "../src/types.ts";
import { validateToolArguments } from "../src/utils/validation.ts";

function createToolCallWithPlainSchema(
	schema: Tool["parameters"],
	value: unknown,
): {
	tool: Tool;
	toolCall: ToolCall;
} {
	const tool: Tool = {
		name: "echo",
		description: "Echo tool",
		parameters: {
			type: "object",
			properties: {
				value: schema,
			},
			required: ["value"],
		} as Tool["parameters"],
	};

	const toolCall: ToolCall = {
		type: "toolCall",
		id: "tool-1",
		name: "echo",
		arguments: { value },
	};

	return { tool, toolCall };
}

describe("validateToolArguments", () => {
	it("still validates when Function constructor is unavailable", () => {
		const originalFunction = globalThis.Function;
		const tool: Tool = {
			name: "echo",
			description: "Echo tool",
			parameters: Type.Object({
				count: Type.Number(),
			}),
		};
		const toolCall: ToolCall = {
			type: "toolCall",
			id: "tool-1",
			name: "echo",
			arguments: { count: "42" as unknown as number },
		};

		globalThis.Function = (() => {
			throw new EvalError("Code generation from strings disallowed for this context");
		}) as unknown as FunctionConstructor;

		try {
			expect(validateToolArguments(tool, toolCall)).toEqual({ count: 42 });
		} finally {
			globalThis.Function = originalFunction;
		}
	});

	it("coerces serialized plain JSON schemas with AJV-compatible primitive rules", () => {
		const passingCases: Array<{
			schema: Tool["parameters"];
			input: unknown;
			expected: unknown;
		}> = [
			{ schema: { type: "number" } as Tool["parameters"], input: "42", expected: 42 },
			{ schema: { type: "number" } as Tool["parameters"], input: true, expected: 1 },
			{ schema: { type: "number" } as Tool["parameters"], input: null, expected: 0 },
			{ schema: { type: "integer" } as Tool["parameters"], input: "42", expected: 42 },
			{ schema: { type: "boolean" } as Tool["parameters"], input: "true", expected: true },
			{ schema: { type: "boolean" } as Tool["parameters"], input: "false", expected: false },
			{ schema: { type: "boolean" } as Tool["parameters"], input: 1, expected: true },
			{ schema: { type: "boolean" } as Tool["parameters"], input: 0, expected: false },
			{ schema: { type: "string" } as Tool["parameters"], input: null, expected: "" },
			{ schema: { type: "string" } as Tool["parameters"], input: true, expected: "true" },
			{ schema: { type: "null" } as Tool["parameters"], input: "", expected: null },
			{ schema: { type: "null" } as Tool["parameters"], input: 0, expected: null },
			{ schema: { type: "null" } as Tool["parameters"], input: false, expected: null },
			{
				schema: { type: ["number", "string"] } as Tool["parameters"],
				input: "1",
				expected: "1",
			},
			{
				schema: { type: ["boolean", "number"] } as Tool["parameters"],
				input: "1",
				expected: 1,
			},
		];

		for (const testCase of passingCases) {
			const { tool, toolCall } = createToolCallWithPlainSchema(testCase.schema, testCase.input);
			expect(validateToolArguments(tool, toolCall)).toEqual({ value: testCase.expected });
		}
	});

	it("treats null as omission for optional non-nullable properties", () => {
		const tool: Tool = {
			name: "echo",
			description: "Echo tool",
			parameters: Type.Object({
				path: Type.String(),
				offset: Type.Optional(Type.Number()),
				nullable: Type.Optional(Type.Union([Type.String(), Type.Null()])),
				metadata: Type.Object({ enabled: Type.Optional(Type.Boolean()) }),
			}),
		};
		const toolCall: ToolCall = {
			type: "toolCall",
			id: "tool-1",
			name: "echo",
			arguments: { path: "file.txt", offset: null, nullable: null, metadata: { enabled: null } },
		};

		expect(validateToolArguments(tool, toolCall)).toEqual({
			path: "file.txt",
			nullable: null,
			metadata: {},
		});
	});

	it("preserves optional nulls whose referenced schema is nullable", () => {
		const tool: Tool = {
			name: "echo",
			description: "Echo tool",
			parameters: {
				type: "object",
				properties: { value: { $ref: "#/$defs/value" } },
				$defs: { value: { anyOf: [{ type: "number" }, { type: "null" }] } },
			} as Tool["parameters"],
		};
		const toolCall: ToolCall = {
			type: "toolCall",
			id: "tool-1",
			name: "echo",
			arguments: { value: null },
		};

		expect(validateToolArguments(tool, toolCall)).toEqual({ value: null });
	});

	it("treats an optional null as omission when the reference resolves non-nullable", () => {
		// The strictifier nullable-wraps optional $ref properties, so the model may
		// answer null for one whose $def does not admit null. Deciding this needs
		// the reference RESOLVED; skipping $ref properties would leave the
		// placeholder in place and fail the call.
		const tool: Tool = {
			name: "echo",
			description: "Echo tool",
			parameters: {
				type: "object",
				properties: { value: { $ref: "#/$defs/value" } },
				required: [],
				$defs: { value: { type: "number" } },
			} as Tool["parameters"],
		};
		const toolCall: ToolCall = {
			type: "toolCall",
			id: "tool-1",
			name: "echo",
			arguments: { value: null },
		};

		expect(validateToolArguments(tool, toolCall)).toEqual({});
	});

	it("resolves references through definitions and escaped pointer segments", () => {
		const tool: Tool = {
			name: "echo",
			description: "Echo tool",
			parameters: {
				type: "object",
				properties: { value: { $ref: "#/definitions/a~1b" } },
				required: [],
				definitions: { "a/b": { type: "number" } },
			} as Tool["parameters"],
		};
		const toolCall: ToolCall = {
			type: "toolCall",
			id: "tool-1",
			name: "echo",
			arguments: { value: null },
		};

		expect(validateToolArguments(tool, toolCall)).toEqual({});
	});

	it("terminates on a self-referencing definition", () => {
		const tool: Tool = {
			name: "echo",
			description: "Echo tool",
			parameters: {
				type: "object",
				properties: { node: { $ref: "#/$defs/node" } },
				required: [],
				$defs: {
					node: { type: "object", properties: { next: { $ref: "#/$defs/node" } } },
				},
			} as Tool["parameters"],
		};
		const toolCall: ToolCall = {
			type: "toolCall",
			id: "tool-1",
			name: "echo",
			arguments: { node: { next: null } },
		};

		expect(validateToolArguments(tool, toolCall)).toEqual({ node: {} });
	});

	it("strips an optional null inside a composed object branch", () => {
		const tool: Tool = {
			name: "echo",
			description: "Echo tool",
			parameters: {
				type: "object",
				anyOf: [{ type: "object", properties: { a: { type: "string" } }, required: [] }],
			} as Tool["parameters"],
		};
		const toolCall: ToolCall = {
			type: "toolCall",
			id: "tool-1",
			name: "echo",
			arguments: { a: null },
		};

		expect(validateToolArguments(tool, toolCall)).toEqual({});
	});

	it("strips a placeholder null that another union arm happens to require", () => {
		// Union arms are mutually exclusive, so `required` must be judged per arm.
		// Aggregating it across arms preserved the placeholder, and the surviving
		// null was then coerced into a fabricated 0 rather than failing.
		const tool: Tool = {
			name: "echo",
			description: "Echo tool",
			parameters: {
				type: "object",
				properties: {
					shape: {
						anyOf: [
							{ type: "object", properties: { x: { type: "number" } }, required: [] },
							{
								type: "object",
								properties: { x: { type: "number" }, y: { type: "number" } },
								required: ["x", "y"],
							},
						],
					},
				},
				required: ["shape"],
			} as Tool["parameters"],
		};
		const toolCall: ToolCall = {
			type: "toolCall",
			id: "tool-1",
			name: "echo",
			arguments: { shape: { x: null } },
		};

		expect(validateToolArguments(tool, toolCall)).toEqual({ shape: {} });
	});

	it("preserves a value that already matches a nullable union arm", () => {
		const tool: Tool = {
			name: "echo",
			description: "Echo tool",
			parameters: Type.Object({
				value: Type.Union([Type.Number(), Type.Null()]),
			}),
		};
		const toolCall: ToolCall = {
			type: "toolCall",
			id: "tool-1",
			name: "echo",
			arguments: { value: null },
		};

		expect(validateToolArguments(tool, toolCall)).toEqual({ value: null });
	});

	it("preserves a value that already matches a oneOf nullable union arm", () => {
		const { tool, toolCall } = createToolCallWithPlainSchema(
			{ oneOf: [{ type: "number" }, { type: "null" }] } as Tool["parameters"],
			null,
		);

		expect(validateToolArguments(tool, toolCall)).toEqual({ value: null });
	});

	it("still coerces nullable unions when the original value does not match any arm", () => {
		const { tool, toolCall } = createToolCallWithPlainSchema(
			{ anyOf: [{ type: "number" }, { type: "null" }] } as Tool["parameters"],
			"42",
		);

		expect(validateToolArguments(tool, toolCall)).toEqual({ value: 42 });
	});

	it("accepts null for nullable array schemas with items", () => {
		const { tool, toolCall } = createToolCallWithPlainSchema(
			{ type: ["array", "null"], items: { type: "string" } } as Tool["parameters"],
			null,
		);
		// The CSP test above selects TypeBox's process-wide interpreted fallback, so exercise the generated validator explicitly.
		const generatedCheck = new Function(Compile(tool.parameters).Code())() as (value: unknown) => boolean;

		expect(generatedCheck(toolCall.arguments)).toBe(true);
		expect(validateToolArguments(tool, toolCall)).toEqual({ value: null });
	});

	it("rejects invalid coercions for serialized plain JSON schemas", () => {
		const failingCases: Array<{
			schema: Tool["parameters"];
			input: unknown;
		}> = [
			{ schema: { type: "boolean" } as Tool["parameters"], input: "1" },
			{ schema: { type: "boolean" } as Tool["parameters"], input: "0" },
			{ schema: { type: "null" } as Tool["parameters"], input: "null" },
			{ schema: { type: "integer" } as Tool["parameters"], input: "42.1" },
		];

		for (const testCase of failingCases) {
			const { tool, toolCall } = createToolCallWithPlainSchema(testCase.schema, testCase.input);
			expect(() => validateToolArguments(tool, toolCall)).toThrow("Validation failed");
		}
	});
});

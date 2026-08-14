import type { Tool } from "../types.ts";
import {
	toAnthropicStrictToolSchema,
	toStrictToolSchema,
	UnstrictifiableSchemaError,
} from "../utils/strict-tool-schema.ts";

interface JsonSchemaObject {
	[key: string]: unknown;
	type?: unknown;
	properties?: Record<string, JsonSchemaObject | undefined>;
	required?: unknown;
}

class UnsupportedStrictJsonSchemaError extends Error {}

function isJsonSchemaObject(value: unknown): value is JsonSchemaObject {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Which provider family's strict subset to target.
 *
 * Anthropic publishes its own strict-schema transformer in its SDK, which is
 * the authoritative definition of what their grammar compiler accepts, so it
 * gets its own dialect rather than sharing the OpenAI-family transform.
 */
export type StrictSchemaDialect = "openai" | "anthropic";

/**
 * Convert a tool schema to the strict subset expected by provider constrained
 * sampling. Throws `UnsupportedStrictJsonSchemaError` when the schema cannot be
 * expressed in that subset; callers decide whether to degrade or fail.
 */
export function makeStrictJsonSchema(
	schema: Tool["parameters"],
	dialect: StrictSchemaDialect = "openai",
): Record<string, unknown> {
	if (!isJsonSchemaObject(schema)) {
		throw new UnsupportedStrictJsonSchemaError("root schema must have type object");
	}
	// Tool parameters specifically must have an object root, in every dialect: a
	// root union has no `properties` for a provider to build a grammar from, and
	// Anthropic's input_schema is required to be an object. This lives here and
	// not in the transforms because the transforms are shared with structured
	// output, where Anthropic does accept a root union.
	if (schema.type !== "object") {
		throw new UnsupportedStrictJsonSchemaError("root schema must have type object");
	}
	try {
		return dialect === "anthropic" ? toAnthropicStrictToolSchema(schema) : toStrictToolSchema(schema);
	} catch (error) {
		if (error instanceof UnstrictifiableSchemaError) throw new UnsupportedStrictJsonSchemaError(error.message);
		throw error;
	}
}

export function getJsonSchemaToolParameters(
	tool: Tool,
	strict: boolean | undefined,
	dialect: StrictSchemaDialect = "openai",
): Tool["parameters"] {
	return (strict === true ? makeStrictJsonSchema(tool.parameters, dialect) : tool.parameters) as Tool["parameters"];
}

export interface GrammarConstrainedSampling {
	format: "lark" | "regex";
	definition: string;
	inputProperty: string;
}

export interface GrammarToolInputJsonBuffer {
	input: string;
	started: boolean;
	closed: boolean;
}

export function getGrammarToolInput(
	toolName: string,
	arguments_: Record<string, unknown>,
	inputProperty: string,
): string {
	const input = arguments_[inputProperty];
	if (typeof input !== "string") {
		throw new Error(`Grammar tool call "${toolName}" requires argument "${inputProperty}" to be a string.`);
	}
	return input;
}

export function appendGrammarToolInputJsonDelta(
	buffer: GrammarToolInputJsonBuffer,
	inputProperty: string,
	nextInput: string,
	close: boolean,
): string | undefined {
	if (buffer.closed) {
		if (close && nextInput === buffer.input) return undefined;
		throw new Error(`grammar tool input for property "${inputProperty}" changed after it was closed`);
	}
	if (!nextInput.startsWith(buffer.input)) {
		throw new Error(`grammar tool input for property "${inputProperty}" changed non-monotonically`);
	}

	const inputDelta = nextInput.slice(buffer.input.length);
	if (!close && inputDelta.length === 0) return undefined;

	let delta = "";
	if (!buffer.started) {
		delta += `{${JSON.stringify(inputProperty)}:"`;
		buffer.started = true;
	}
	delta += JSON.stringify(inputDelta).slice(1, -1);
	buffer.input = nextInput;

	if (close) {
		delta += '"}';
		buffer.closed = true;
	}
	return delta;
}

function inferGrammarInputProperty(tool: Tool): string {
	const schema = tool.parameters as JsonSchemaObject;
	if (schema.type !== "object") {
		throw new Error("grammar constrained sampling requires an object parameter schema");
	}
	if (!Array.isArray(schema.required) || schema.required.length !== 1 || typeof schema.required[0] !== "string") {
		throw new Error("grammar constrained sampling requires exactly one required string property");
	}

	const inputProperty = schema.required[0];
	if (!schema.properties?.[inputProperty]) {
		throw new Error(`grammar constrained sampling requires a properties entry for ${inputProperty}`);
	}
	if (schema.properties[inputProperty]?.type !== "string") {
		throw new Error(`grammar constrained sampling property ${inputProperty} must have type string`);
	}
	return inputProperty;
}

export function resolveJsonSchemaStrictSampling(
	tool: Tool,
	supportsStrictMode: boolean,
	dialect: StrictSchemaDialect = "openai",
): boolean | undefined {
	const config = tool.constrainedSampling;
	if (!config || config.type !== "json_schema") return undefined;

	if (supportsStrictMode) {
		try {
			// Probe with the SAME dialect the caller will use to produce the schema:
			// a schema one family accepts is not necessarily one the other accepts,
			// and both results are cached per schema identity so this is not extra work.
			makeStrictJsonSchema(tool.parameters, dialect);
			return true;
		} catch (error) {
			if (!(error instanceof UnsupportedStrictJsonSchemaError)) throw error;
			if (config.strict !== "require") return undefined;
			throw new Error(`Tool "${tool.name}" requires JSON-schema constrained sampling, but ${error.message}.`);
		}
	}
	if (config.strict === "require") {
		throw new Error(
			`Tool "${tool.name}" requires JSON-schema constrained sampling, but strict tools are unsupported.`,
		);
	}
	return undefined;
}

export function resolveGrammarConstrainedSampling(
	tool: Tool,
	supportsOpenAIGrammarTools: boolean,
): GrammarConstrainedSampling | undefined {
	const config = tool.constrainedSampling;
	if (!config || config.type !== "grammar") {
		return undefined;
	}

	if (!supportsOpenAIGrammarTools) {
		return undefined;
	}

	const larkDefinition = config.variants.openai_lark;
	const regexDefinition = config.variants.openai_regex;
	const hasLarkDefinition = typeof larkDefinition === "string" && larkDefinition.trim().length > 0;
	const hasRegexDefinition = typeof regexDefinition === "string" && regexDefinition.trim().length > 0;
	if (!hasLarkDefinition && !hasRegexDefinition) {
		throw new Error(
			`Tool "${tool.name}" cannot use grammar constrained sampling: no supported grammar variant was provided.`,
		);
	}

	try {
		return {
			format: hasLarkDefinition ? "lark" : "regex",
			definition: hasLarkDefinition ? larkDefinition : regexDefinition!,
			inputProperty: inferGrammarInputProperty(tool),
		};
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error);
		throw new Error(`Tool "${tool.name}" cannot use grammar constrained sampling: ${message}.`);
	}
}

export function createGrammarToolInputProperties(
	tools: Tool[] | undefined,
	supportsOpenAIGrammarTools: boolean,
): ReadonlyMap<string, string> {
	const properties = new Map<string, string>();
	for (const tool of tools ?? []) {
		const grammar = resolveGrammarConstrainedSampling(tool, supportsOpenAIGrammarTools);
		if (grammar) {
			properties.set(tool.name, grammar.inputProperty);
		}
	}
	return properties;
}

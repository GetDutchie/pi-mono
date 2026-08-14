import { Compile } from "typebox/compile";
import type { TLocalizedValidationError } from "typebox/error";
import { Value } from "typebox/value";
import type { Tool, ToolCall } from "../types.ts";

const validatorCache = new WeakMap<object, ReturnType<typeof Compile>>();
const TYPEBOX_KIND = Symbol.for("TypeBox.Kind");

interface JsonSchemaObject {
	type?: string | string[];
	properties?: Record<string, JsonSchemaObject>;
	required?: string[];
	items?: JsonSchemaObject | JsonSchemaObject[];
	additionalProperties?: boolean | JsonSchemaObject;
	allOf?: JsonSchemaObject[];
	anyOf?: JsonSchemaObject[];
	oneOf?: JsonSchemaObject[];
	$ref?: string;
	$defs?: Record<string, JsonSchemaObject>;
	definitions?: Record<string, JsonSchemaObject>;
	const?: unknown;
	enum?: unknown[];
}

function getSchemaTypes(schema: JsonSchemaObject): string[] {
	if (typeof schema.type === "string") {
		return [schema.type];
	}
	if (Array.isArray(schema.type)) {
		return schema.type.filter((type): type is string => typeof type === "string");
	}
	return [];
}

function matchesJsonType(value: unknown, type: string): boolean {
	switch (type) {
		case "number":
			return typeof value === "number";
		case "integer":
			return typeof value === "number" && Number.isInteger(value);
		case "boolean":
			return typeof value === "boolean";
		case "string":
			return typeof value === "string";
		case "null":
			return value === null;
		case "array":
			return Array.isArray(value);
		case "object":
			return typeof value === "object" && value !== null && !Array.isArray(value);
		default:
			return false;
	}
}

function getSubSchemaValidator(schema: JsonSchemaObject): ReturnType<typeof Compile> | undefined {
	try {
		return getValidator(schema as Tool["parameters"]);
	} catch {
		return undefined;
	}
}

function coercePrimitiveByType(value: unknown, type: string): unknown {
	switch (type) {
		case "number": {
			if (value === null) {
				return 0;
			}
			if (typeof value === "string" && value.trim() !== "") {
				const parsed = Number(value);
				if (Number.isFinite(parsed)) {
					return parsed;
				}
			}
			if (typeof value === "boolean") {
				return value ? 1 : 0;
			}
			return value;
		}
		case "integer": {
			if (value === null) {
				return 0;
			}
			if (typeof value === "string" && value.trim() !== "") {
				const parsed = Number(value);
				if (Number.isInteger(parsed)) {
					return parsed;
				}
			}
			if (typeof value === "boolean") {
				return value ? 1 : 0;
			}
			return value;
		}
		case "boolean": {
			if (value === null) {
				return false;
			}
			if (typeof value === "string") {
				if (value === "true") {
					return true;
				}
				if (value === "false") {
					return false;
				}
			}
			if (typeof value === "number") {
				if (value === 1) {
					return true;
				}
				if (value === 0) {
					return false;
				}
			}
			return value;
		}
		case "string": {
			if (value === null) {
				return "";
			}
			if (typeof value === "number" || typeof value === "boolean") {
				return String(value);
			}
			return value;
		}
		case "null": {
			if (value === "" || value === 0 || value === false) {
				return null;
			}
			return value;
		}
		default:
			return value;
	}
}

function applySchemaObjectCoercion(value: Record<string, unknown>, schema: JsonSchemaObject): void {
	const properties = schema.properties;
	const definedKeys = new Set<string>(properties ? Object.keys(properties) : []);

	if (properties) {
		for (const [key, propertySchema] of Object.entries(properties)) {
			if (!(key in value)) {
				continue;
			}
			value[key] = coerceWithJsonSchema(value[key], propertySchema);
		}
	}

	if (schema.additionalProperties && typeof schema.additionalProperties === "object") {
		for (const [key, propertyValue] of Object.entries(value)) {
			if (definedKeys.has(key)) {
				continue;
			}
			value[key] = coerceWithJsonSchema(propertyValue, schema.additionalProperties);
		}
	}
}

function applySchemaArrayCoercion(value: unknown[], schema: JsonSchemaObject): void {
	if (Array.isArray(schema.items)) {
		for (let index = 0; index < value.length; index++) {
			const itemSchema = schema.items[index];
			if (!itemSchema) {
				continue;
			}
			value[index] = coerceWithJsonSchema(value[index], itemSchema);
		}
		return;
	}

	if (schema.items && typeof schema.items === "object") {
		for (let index = 0; index < value.length; index++) {
			value[index] = coerceWithJsonSchema(value[index], schema.items);
		}
	}
}

function coerceWithUnionSchema(value: unknown, schemas: JsonSchemaObject[]): unknown {
	for (const schema of schemas) {
		const validator = getSubSchemaValidator(schema);
		if (validator?.Check(value)) {
			return value;
		}
	}

	for (const schema of schemas) {
		const candidate = structuredClone(value);
		const coerced = coerceWithJsonSchema(candidate, schema);
		const validator = getSubSchemaValidator(schema);
		if (validator?.Check(coerced)) {
			return coerced;
		}
	}
	return value;
}

function coerceWithJsonSchema(value: unknown, schema: JsonSchemaObject): unknown {
	let nextValue = value;

	if (Array.isArray(schema.allOf)) {
		for (const nested of schema.allOf) {
			nextValue = coerceWithJsonSchema(nextValue, nested);
		}
	}

	if (Array.isArray(schema.anyOf)) {
		nextValue = coerceWithUnionSchema(nextValue, schema.anyOf);
	}

	if (Array.isArray(schema.oneOf)) {
		nextValue = coerceWithUnionSchema(nextValue, schema.oneOf);
	}

	const schemaTypes = getSchemaTypes(schema);
	const matchesUnionMember =
		schemaTypes.length > 1 && schemaTypes.some((schemaType) => matchesJsonType(nextValue, schemaType));
	if (schemaTypes.length > 0 && !matchesUnionMember) {
		for (const schemaType of schemaTypes) {
			const candidate = coercePrimitiveByType(nextValue, schemaType);
			if (candidate !== nextValue) {
				nextValue = candidate;
				break;
			}
		}
	}

	if (
		schemaTypes.includes("object") &&
		typeof nextValue === "object" &&
		nextValue !== null &&
		!Array.isArray(nextValue)
	) {
		applySchemaObjectCoercion(nextValue as Record<string, unknown>, schema);
	}

	if (schemaTypes.includes("array") && Array.isArray(nextValue)) {
		applySchemaArrayCoercion(nextValue, schema);
	}

	return nextValue;
}

/**
 * Strict-mode tool schemas (see utils/strict-tool-schema.ts) cannot express
 * "optional": every property must appear in `required`, so previously-optional
 * ones are made nullable and the model emits `null` to mean "absent". This is
 * the inbound half of that bargain: drop those placeholder nulls before the
 * arguments are validated against the ORIGINAL schema, where the property is
 * optional and non-nullable.
 *
 * The hard case is `$ref`. Whether a `null` is a placeholder or a value the
 * schema genuinely permits lives at the far end of the reference, so this
 * RESOLVES local references before deciding. Guessing structurally would
 * delete legitimate nulls for properties whose `$def` is nullable; refusing to
 * strip `$ref` properties at all would leave placeholder nulls in place and
 * fail the call. References that cannot be resolved (external URLs, unknown
 * pointers, cycles) are treated as nullable, i.e. the null is preserved.
 *
 * Stripping is safe even for tools that were never strictified: a `null` for an
 * optional property that does not admit null would have failed validation
 * anyway, so treating it as an omission is strictly an improvement.
 */
function unescapeJsonPointerSegment(segment: string): string {
	return segment.replace(/~1/g, "/").replace(/~0/g, "~");
}

/** Follow a local `$ref` chain to the schema it names, or undefined if it cannot be resolved. */
function resolveSchemaRef(root: JsonSchemaObject, schema: JsonSchemaObject | undefined): JsonSchemaObject | undefined {
	let current = schema;
	const seen = new Set<string>();
	while (current && typeof current.$ref === "string") {
		const ref = current.$ref;
		if (seen.has(ref)) return undefined;
		seen.add(ref);

		const match = /^#\/(\$defs|definitions)\/(.+)$/.exec(ref);
		if (!match) return undefined;
		const pool = match[1] === "$defs" ? root.$defs : root.definitions;
		const target = pool?.[unescapeJsonPointerSegment(match[2])];
		if (!target || typeof target !== "object") return undefined;
		current = target;
	}
	return current;
}

function schemaAllowsNull(
	root: JsonSchemaObject,
	schema: JsonSchemaObject | undefined,
	visited: Set<JsonSchemaObject> = new Set(),
): boolean {
	const resolved = resolveSchemaRef(root, schema);
	// An absent or unresolvable schema tells us nothing, so keep the null.
	if (!resolved) return true;
	if (visited.has(resolved)) return true;
	visited.add(resolved);

	const type = resolved.type;
	if (type === "null") return true;
	if (Array.isArray(type) && type.includes("null")) return true;
	if ("const" in resolved && resolved.const === null) return true;
	if (Array.isArray(resolved.enum) && resolved.enum.includes(null)) return true;

	for (const union of [resolved.anyOf, resolved.oneOf]) {
		if (Array.isArray(union) && union.some((member) => schemaAllowsNull(root, member, visited))) return true;
	}
	// `null` must satisfy every allOf branch to be admissible.
	if (Array.isArray(resolved.allOf) && resolved.allOf.length > 0) {
		return resolved.allOf.every((member) => schemaAllowsNull(root, member, visited));
	}
	return false;
}

interface ObjectSchemaView {
	props: Record<string, JsonSchemaObject>;
	required: Set<string>;
}

/**
 * Every object-shaped view of a schema: the node itself plus any object
 * branches under allOf/anyOf/oneOf. The strictifier recurses into compositions,
 * so its placeholder nulls can appear inside them too.
 */
function collectObjectViews(root: JsonSchemaObject, schema: JsonSchemaObject | undefined): ObjectSchemaView[] {
	const views: ObjectSchemaView[] = [];
	const visited = new Set<JsonSchemaObject>();
	const visit = (node: JsonSchemaObject | undefined): void => {
		const resolved = resolveSchemaRef(root, node);
		if (!resolved || visited.has(resolved)) return;
		visited.add(resolved);
		if (resolved.properties && typeof resolved.properties === "object") {
			views.push({
				props: resolved.properties,
				required: new Set(Array.isArray(resolved.required) ? resolved.required : []),
			});
		}
		for (const union of [resolved.allOf, resolved.anyOf, resolved.oneOf]) {
			if (Array.isArray(union)) for (const member of union) visit(member);
		}
	};
	visit(schema);
	return views;
}

function stripStrictModeNulls(root: JsonSchemaObject, value: unknown, schema: JsonSchemaObject | undefined): void {
	if (value === null || typeof value !== "object") return;
	const resolved = resolveSchemaRef(root, schema);
	if (!resolved) return;

	if (Array.isArray(value)) {
		const items = resolved.items;
		if (Array.isArray(items)) {
			for (const [index, entry] of value.entries()) stripStrictModeNulls(root, entry, items[index]);
		} else if (items) {
			for (const entry of value) stripStrictModeNulls(root, entry, items);
		}
		return;
	}

	const views = collectObjectViews(root, resolved);
	if (views.length === 0) return;
	const object = value as Record<string, unknown>;
	for (const key of Object.keys(object)) {
		const mentioned = views.filter((view) => key in view.props);
		if (mentioned.length === 0) continue;
		if (object[key] === null) {
			// Strip only when NO view requires the key and NO view admits null for
			// it, i.e. the null is unambiguously a strict-mode placeholder.
			const anyRequires = views.some((view) => view.required.has(key));
			const anyAllowsNull = mentioned.some((view) => schemaAllowsNull(root, view.props[key]));
			if (!anyRequires && !anyAllowsNull) delete object[key];
			continue;
		}
		for (const view of mentioned) stripStrictModeNulls(root, object[key], view.props[key]);
	}
}

function getValidator(schema: Tool["parameters"]): ReturnType<typeof Compile> {
	const key = schema as object;
	const cached = validatorCache.get(key);
	if (cached) {
		return cached;
	}
	const validator = Compile(schema);
	validatorCache.set(key, validator);
	return validator;
}

function formatValidationPath(error: TLocalizedValidationError): string {
	if (error.keyword === "required") {
		const requiredProperties = (error.params as { requiredProperties?: string[] }).requiredProperties;
		const requiredProperty = requiredProperties?.[0];
		if (requiredProperty) {
			const basePath = error.instancePath.replace(/^\//, "").replace(/\//g, ".");
			return basePath ? `${basePath}.${requiredProperty}` : requiredProperty;
		}
	}
	const path = error.instancePath.replace(/^\//, "").replace(/\//g, ".");
	return path || "root";
}

/**
 * Finds a tool by name and validates the tool call arguments against its TypeBox schema
 * @param tools Array of tool definitions
 * @param toolCall The tool call from the LLM
 * @returns The validated arguments
 * @throws Error if tool is not found or validation fails
 */
export function validateToolCall(tools: Tool[], toolCall: ToolCall): any {
	const tool = tools.find((t) => t.name === toolCall.name);
	if (!tool) {
		throw new Error(`Tool "${toolCall.name}" not found`);
	}
	return validateToolArguments(tool, toolCall);
}

/**
 * Validates tool call arguments against the tool's TypeBox schema
 * @param tool The tool definition with TypeBox schema
 * @param toolCall The tool call from the LLM
 * @returns The validated (and potentially coerced) arguments
 * @throws Error with formatted message if validation fails
 */
export function validateToolArguments(tool: Tool, toolCall: ToolCall): any {
	const args = structuredClone(toolCall.arguments);
	const parameters = tool.parameters as JsonSchemaObject;
	stripStrictModeNulls(parameters, args, parameters);
	Value.Convert(tool.parameters, args);

	const validator = getValidator(tool.parameters);
	if (!Object.getOwnPropertySymbols(tool.parameters).includes(TYPEBOX_KIND)) {
		const coerced = coerceWithJsonSchema(args, tool.parameters as JsonSchemaObject);
		if (coerced !== args) {
			if (typeof args === "object" && args !== null && typeof coerced === "object" && coerced !== null) {
				for (const key of Object.keys(args)) {
					delete args[key];
				}
				Object.assign(args, coerced);
			} else {
				return validator.Check(coerced) ? coerced : args;
			}
		}
	}

	if (validator.Check(args)) {
		return args;
	}

	const errors =
		validator
			.Errors(args)
			.map((error) => `  - ${formatValidationPath(error)}: ${error.message}`)
			.join("\n") || "Unknown validation error";

	const errorMessage = `Validation failed for tool "${toolCall.name}":\n${errors}\n\nReceived arguments:\n${JSON.stringify(toolCall.arguments, null, 2)}`;

	throw new Error(errorMessage);
}

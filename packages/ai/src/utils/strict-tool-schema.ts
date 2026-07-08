/**
 * Strict-mode tool schema transformation.
 *
 * OpenAI-family strict mode (`tools[].strict: true`) makes the provider
 * logit-mask tool-call arguments against the schema grammar — the model
 * physically cannot emit arguments that violate the schema, which replaces
 * the validate-and-reprompt fallback for schema conformance.
 *
 * Strict mode requires a restricted JSON Schema subset:
 *   - every object must set `additionalProperties: false`
 *   - every property must be listed in `required` (optionals are expressed
 *     as nullable: the model emits `null` for "absent", and
 *     `validateToolArguments` strips those nulls back off before validating
 *     against the ORIGINAL schema)
 *   - several validation keywords are unsupported and must be stripped
 *     (they remain enforced post-hoc by the original-schema validation, so
 *     a violation still falls back to the reprompt loop — grammar handles
 *     structure/types/enums, the loop handles residual constraints)
 *
 * `strictToolSchema` returns the transformed schema, or `null` when the
 * schema cannot be expressed in the strict subset (e.g. it explicitly
 * relies on `additionalProperties`/`patternProperties` passthrough) — in
 * that case the caller sends the tool WITHOUT strict and the pre-existing
 * validate-and-reprompt loop remains that tool's enforcement.
 */

// Keywords OpenAI strict mode rejects. They are advisory-to-the-model only in
// strict requests; runtime validation still enforces them from the original
// schema.
const UNSUPPORTED_KEYWORDS = new Set([
	"format",
	"pattern",
	"minLength",
	"maxLength",
	"minimum",
	"maximum",
	"exclusiveMinimum",
	"exclusiveMaximum",
	"multipleOf",
	"minItems",
	"maxItems",
	"uniqueItems",
	"contains",
	"minContains",
	"maxContains",
	"minProperties",
	"maxProperties",
	"default",
	"examples",
	"contentEncoding",
	"contentMediaType",
	"deprecated",
	"readOnly",
	"writeOnly",
]);

// Structural keywords strict mode cannot express — their presence makes the
// whole tool unstrictifiable (send unstrict, keep the reprompt loop).
const UNSTRICTIFIABLE_KEYWORDS = new Set([
	"patternProperties",
	"propertyNames",
	"unevaluatedProperties",
	"unevaluatedItems",
	"dependentRequired",
	"dependentSchemas",
	"if",
	"then",
	"else",
	"not",
]);

class Unstrictifiable extends Error {}

function transformNode(node: unknown): unknown {
	if (node === null || typeof node !== "object") return node;
	if (Array.isArray(node)) return node.map(transformNode);

	const obj = node as Record<string, unknown>;
	const out: Record<string, unknown> = {};

	for (const [key, value] of Object.entries(obj)) {
		if (UNSUPPORTED_KEYWORDS.has(key)) continue;
		if (UNSTRICTIFIABLE_KEYWORDS.has(key)) throw new Unstrictifiable(key);
		if (key === "additionalProperties") {
			// Explicit passthrough objects cannot be expressed in strict mode.
			if (value !== false) throw new Unstrictifiable("additionalProperties");
			continue; // re-added below
		}
		if (key === "oneOf") {
			// strict mode supports anyOf but not oneOf
			out.anyOf = (value as unknown[]).map(transformNode);
			continue;
		}
		if (key === "properties" && value !== null && typeof value === "object") {
			const props: Record<string, unknown> = {};
			for (const [pk, pv] of Object.entries(value as Record<string, unknown>)) {
				props[pk] = transformNode(pv);
			}
			out.properties = props;
			continue;
		}
		if (key === "items" || key === "anyOf" || key === "allOf" || key === "$defs" || key === "definitions") {
			out[key] = transformNode(value);
			continue;
		}
		out[key] = value;
	}

	// Object nodes: strict mode mandates additionalProperties:false and
	// required = ALL keys. Previously-optional properties become nullable so
	// the model can express absence; validateToolArguments strips those nulls.
	if (out.type === "object" && out.properties !== null && typeof out.properties === "object") {
		out.additionalProperties = false;
		const props = out.properties as Record<string, unknown>;
		const allKeys = Object.keys(props);
		const origRequired = new Set(Array.isArray(out.required) ? (out.required as string[]) : []);
		out.required = allKeys;
		for (const pk of allKeys) {
			if (origRequired.has(pk)) continue;
			const pv = props[pk];
			if (pv === null || typeof pv !== "object" || Array.isArray(pv)) continue;
			props[pk] = makeNullable(pv as Record<string, unknown>);
		}
	}

	return out;
}

function makeNullable(prop: Record<string, unknown>): Record<string, unknown> {
	const t = prop.type;
	if (Array.isArray(t)) {
		return t.includes("null") ? prop : { ...prop, type: [...t, "null"] };
	}
	if (typeof t === "string") {
		return t === "null" ? prop : { ...prop, type: [t, "null"] };
	}
	if (Array.isArray(prop.anyOf)) {
		const anyOf = prop.anyOf as Record<string, unknown>[];
		if (anyOf.some((m) => m?.type === "null")) return prop;
		return { ...prop, anyOf: [...anyOf, { type: "null" }] };
	}
	// No type info (bare schema) — leave as-is; strict mode treats it as any.
	return prop;
}

const strictCache = new WeakMap<object, Record<string, unknown> | null>();

/**
 * Transform a tool parameter schema into the OpenAI strict-mode subset.
 * Returns `null` when the schema cannot be strictified — the caller must
 * then send the tool without strict mode (reprompt-loop enforcement).
 * Results are cached per schema object identity.
 */
export function strictToolSchema(schema: object): Record<string, unknown> | null {
	const cached = strictCache.get(schema);
	if (cached !== undefined) return cached;
	let result: Record<string, unknown> | null;
	try {
		// JSON round-trip drops TypeBox symbol metadata providers reject.
		const raw = JSON.parse(JSON.stringify(schema)) as unknown;
		result = transformNode(raw) as Record<string, unknown>;
	} catch {
		result = null;
	}
	strictCache.set(schema, result);
	return result;
}

// ---------------------------------------------------------------------------
// Anthropic strict tool use (GA since early 2026)
// ---------------------------------------------------------------------------

/**
 * Anthropic's strict tool use requires `additionalProperties: false` on every
 * object and an explicit `required` list, but — unlike OpenAI — does NOT
 * require every property to be listed in `required`, so optional properties
 * stay optional (no nullable-optional transformation, no null-stripping on
 * the way back). All other keywords pass through to Anthropic's grammar
 * compiler untouched.
 */
function anthropicTransformNode(node: unknown): unknown {
	if (node === null || typeof node !== "object") return node;
	if (Array.isArray(node)) return node.map(anthropicTransformNode);

	const obj = node as Record<string, unknown>;
	const out: Record<string, unknown> = {};

	for (const [key, value] of Object.entries(obj)) {
		if (key === "additionalProperties") {
			// Explicit passthrough objects cannot be expressed in strict mode.
			if (value !== false) throw new Unstrictifiable("additionalProperties");
			continue; // re-added below
		}
		if (key === "properties" && value !== null && typeof value === "object") {
			const props: Record<string, unknown> = {};
			for (const [pk, pv] of Object.entries(value as Record<string, unknown>)) {
				props[pk] = anthropicTransformNode(pv);
			}
			out.properties = props;
			continue;
		}
		if (
			key === "items" ||
			key === "anyOf" ||
			key === "allOf" ||
			key === "oneOf" ||
			key === "$defs" ||
			key === "definitions"
		) {
			out[key] = anthropicTransformNode(value);
			continue;
		}
		out[key] = value;
	}

	if (out.type === "object" && out.properties !== null && typeof out.properties === "object") {
		out.additionalProperties = false;
		if (!Array.isArray(out.required)) out.required = [];
	}

	return out;
}

const anthropicStrictCache = new WeakMap<object, Record<string, unknown> | null>();

/**
 * Transform a tool parameter schema into Anthropic's strict-tool-use shape.
 * Returns `null` when the schema cannot be strictified (explicit
 * additionalProperties passthrough) — that tool is sent without strict and
 * keeps the reprompt loop. Results are cached per schema object identity.
 */
export function anthropicStrictToolSchema(schema: object): Record<string, unknown> | null {
	const cached = anthropicStrictCache.get(schema);
	if (cached !== undefined) return cached;
	let result: Record<string, unknown> | null;
	try {
		const raw = JSON.parse(JSON.stringify(schema)) as unknown;
		result = anthropicTransformNode(raw) as Record<string, unknown>;
	} catch {
		result = null;
	}
	anthropicStrictCache.set(schema, result);
	return result;
}

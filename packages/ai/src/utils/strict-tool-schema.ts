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
 *
 * `toStrictToolSchema` is the same transform but throws
 * `UnstrictifiableSchemaError` instead of returning `null`, so a tool that
 * declares `constrainedSampling.strict: "require"` can report WHY strict mode
 * was unavailable rather than silently degrading.
 *
 * `$ref`/`$defs` survive the transform: both OpenAI and Anthropic strict modes
 * resolve local references, and TypeBox emits them for any reused sub-schema.
 * Optional `$ref` properties are nullable-wrapped like any other, which is why
 * the inbound null-stripper in `utils/validation.ts` must RESOLVE references
 * before deciding whether a `null` is a strict-mode placeholder.
 */

import { transformJSONSchema } from "@anthropic-ai/sdk/lib/transform-json-schema";

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

function unstrictifiableReason(error: Unstrictifiable): string {
	return `${error.message} is unsupported in the strict schema subset`;
}

/**
 * Failures that are not our own structural rejection (a provider SDK
 * transformer throwing, malformed JSON) still mean "send this tool unstrict",
 * but the reason is worth surfacing for `strict: "require"`.
 */
function describeFailure(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}

/**
 * Tool parameters and structured-output schemas must be object schemas: every
 * provider's strict surface rejects a bare primitive or array root.
 */
function assertObjectRoot(node: unknown): void {
	if (node === null || typeof node !== "object" || Array.isArray(node)) {
		throw new Unstrictifiable("a non-object root schema");
	}
	if ((node as Record<string, unknown>).type !== "object") {
		throw new Unstrictifiable('a root schema without type "object"');
	}
}

function unescapeJsonPointerSegment(segment: string): string {
	return segment.replace(/~1/g, "/").replace(/~0/g, "~");
}

/**
 * Every `$ref` in a strict schema must be a LOCAL pointer into this document's
 * definition pool, must resolve, and must not participate in a cycle.
 *
 * A provider's grammar compiler only ever sees the document we send it: an
 * external `$ref` is unfetchable, a dangling pointer is unresolvable, and
 * recursive reference graphs are not accepted across the strict subsets we
 * target (notably Amazon Bedrock). Any of those would be rejected at admission
 * time for the whole request, so such a schema is unstrictifiable and the tool
 * must be sent unstrict instead.
 */
function assertResolvableLocalRefs(root: Record<string, unknown>): void {
	const resolveRef = (ref: unknown): Record<string, unknown> => {
		if (typeof ref !== "string") throw new Unstrictifiable("a non-string $ref");
		const match = /^#\/(\$defs|definitions)\/(.+)$/.exec(ref);
		if (!match) throw new Unstrictifiable(`a non-local $ref (${ref})`);
		const pool = root[match[1]];
		const target =
			pool && typeof pool === "object" && !Array.isArray(pool)
				? (pool as Record<string, unknown>)[unescapeJsonPointerSegment(match[2])]
				: undefined;
		if (!target || typeof target !== "object" || Array.isArray(target)) {
			throw new Unstrictifiable(`an unresolvable $ref (${ref})`);
		}
		return target as Record<string, unknown>;
	};

	// Grey/black DFS over the reference graph: grey means "on the current
	// resolution chain", so re-entering it is a cycle, and black means "already
	// proven acyclic", which keeps shared definitions linear rather than
	// re-expanding them at every use site.
	const grey = new Set<string>();
	const black = new Set<string>();
	const walk = (node: unknown): void => {
		if (node === null || typeof node !== "object") return;
		if (Array.isArray(node)) {
			for (const entry of node) walk(entry);
			return;
		}
		for (const [key, value] of Object.entries(node as Record<string, unknown>)) {
			if (key !== "$ref") {
				walk(value);
				continue;
			}
			const ref = value as string;
			if (grey.has(ref)) throw new Unstrictifiable(`a recursive $ref (${ref})`);
			const target = resolveRef(ref);
			if (black.has(ref)) continue;
			grey.add(ref);
			walk(target);
			grey.delete(ref);
			black.add(ref);
		}
	};
	walk(root);
}

/**
 * Thrown by the throwing transform variants when a schema cannot be expressed
 * in a provider's strict subset. Callers that want per-tool fallback use the
 * `null`-returning variants instead.
 */
export class UnstrictifiableSchemaError extends Error {}

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
		if (key === "items" && Array.isArray(value)) {
			// Tuple-form items cannot be expressed in the strict subset — sending
			// them strict gets the whole request 400'd at admission time.
			throw new Unstrictifiable("tuple items");
		}
		if (key === "allOf") {
			// OpenAI's strict subset does not accept allOf — sending it strict
			// admission-fails the whole request. Fall back to the reprompt loop.
			throw new Unstrictifiable("allOf");
		}
		if (key === "$defs" || key === "definitions") {
			// A definition pool is a MAP of schemas, not a schema. Strictify each
			// entry: a $def reached through $ref must satisfy the strict subset
			// too, or the provider rejects the whole schema.
			if (value === null || typeof value !== "object" || Array.isArray(value)) {
				throw new Unstrictifiable(`a non-object ${key} pool`);
			}
			const defs: Record<string, unknown> = {};
			for (const [name, def] of Object.entries(value as Record<string, unknown>)) {
				defs[name] = transformNode(def);
			}
			out[key] = defs;
			continue;
		}
		if (key === "items" || key === "anyOf") {
			out[key] = transformNode(value);
			continue;
		}
		out[key] = value;
	}

	// Object nodes: strict mode mandates additionalProperties:false and
	// required = ALL keys. Previously-optional properties become nullable so
	// the model can express absence; validateToolArguments strips those nulls.
	if (out.type === "object") {
		out.additionalProperties = false;
		if (out.properties === null || typeof out.properties !== "object") out.properties = {};
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
	// No direct type ($ref, bare schema): wrap in an anyOf with null so
	// the model can still express absence. Making such a property required
	// WITHOUT a null escape would silently change the tool's input contract
	// (the model would be grammar-forced to invent a value).
	return { anyOf: [prop, { type: "null" }] };
}

type StrictResult = { ok: Record<string, unknown> } | { reason: string };

const strictCache = new WeakMap<object, StrictResult>();

function computeStrict(schema: object): StrictResult {
	try {
		// JSON round-trip drops TypeBox symbol metadata providers reject.
		const raw = JSON.parse(JSON.stringify(schema)) as unknown;
		assertObjectRoot(raw);
		assertResolvableLocalRefs(raw as Record<string, unknown>);
		return { ok: transformNode(raw) as Record<string, unknown> };
	} catch (error) {
		return { reason: error instanceof Unstrictifiable ? unstrictifiableReason(error) : describeFailure(error) };
	}
}

/**
 * Transform a tool parameter schema into the OpenAI strict-mode subset,
 * throwing `UnstrictifiableSchemaError` when the schema cannot be expressed in
 * it. Results (including failures) are cached per schema object identity.
 */
export function toStrictToolSchema(schema: object): Record<string, unknown> {
	let result = strictCache.get(schema);
	if (result === undefined) {
		result = computeStrict(schema);
		strictCache.set(schema, result);
	}
	if ("reason" in result) throw new UnstrictifiableSchemaError(result.reason);
	return result.ok;
}

/**
 * Transform a tool parameter schema into the OpenAI strict-mode subset.
 * Returns `null` when the schema cannot be strictified — the caller must
 * then send the tool without strict mode (reprompt-loop enforcement).
 */
export function strictToolSchema(schema: object): Record<string, unknown> | null {
	try {
		return toStrictToolSchema(schema);
	} catch (error) {
		if (error instanceof UnstrictifiableSchemaError) return null;
		throw error;
	}
}

// ---------------------------------------------------------------------------
// Anthropic strict tool use (GA since early 2026)
// ---------------------------------------------------------------------------

/**
 * Anthropic strict tool use, via the Anthropic SDK's OWN strict-schema
 * transformer (`@anthropic-ai/sdk/lib/transform-json-schema`) — the
 * authoritative definition of what their grammar compiler accepts. It
 * whitelists supported fields, forces `additionalProperties: false`,
 * converts oneOf→anyOf, and moves unsupported keywords (including `enum`,
 * `pattern`, numeric bounds) into `description` — those stay enforced
 * post-hoc by original-schema validation (the reprompt loop remains their
 * fallback), while structure and types are grammar-enforced.
 *
 * Structural pre-check: keywords whose SEMANTICS cannot survive the strict
 * transform (passthrough/conditional shapes) make the tool unstrictifiable
 * entirely — forcing additionalProperties:false while dropping
 * patternProperties would silently break a passthrough tool, not just relax
 * its enforcement. The SDK transformer throwing (e.g. tuple items, typeless
 * nodes) also falls back to unstrict.
 */
function assertAnthropicStrictifiable(node: unknown): void {
	if (node === null || typeof node !== "object") return;
	if (Array.isArray(node)) {
		for (const entry of node) assertAnthropicStrictifiable(entry);
		return;
	}
	const obj = node as Record<string, unknown>;
	for (const [key, value] of Object.entries(obj)) {
		if (UNSTRICTIFIABLE_KEYWORDS.has(key)) throw new Unstrictifiable(key);
		if (key === "additionalProperties" && value !== false) throw new Unstrictifiable("additionalProperties");
		assertAnthropicStrictifiable(value);
	}
}

const anthropicStrictCache = new WeakMap<object, StrictResult>();

function computeAnthropicStrict(schema: object): StrictResult {
	try {
		const raw = JSON.parse(JSON.stringify(schema)) as unknown;
		assertObjectRoot(raw);
		assertResolvableLocalRefs(raw as Record<string, unknown>);
		assertAnthropicStrictifiable(raw);
		return { ok: transformJSONSchema(raw as Parameters<typeof transformJSONSchema>[0]) as Record<string, unknown> };
	} catch (error) {
		return { reason: error instanceof Unstrictifiable ? unstrictifiableReason(error) : describeFailure(error) };
	}
}

/**
 * Transform a tool parameter schema into Anthropic's strict-tool-use shape,
 * throwing `UnstrictifiableSchemaError` when the schema cannot be expressed in
 * it. Results (including failures) are cached per schema object identity.
 */
export function toAnthropicStrictToolSchema(schema: object): Record<string, unknown> {
	let result = anthropicStrictCache.get(schema);
	if (result === undefined) {
		result = computeAnthropicStrict(schema);
		anthropicStrictCache.set(schema, result);
	}
	if ("reason" in result) throw new UnstrictifiableSchemaError(result.reason);
	return result.ok;
}

/**
 * Transform a tool parameter schema into Anthropic's strict-tool-use shape.
 * Returns `null` when the schema cannot be strictified — that tool is sent
 * without strict and keeps the reprompt loop.
 */
export function anthropicStrictToolSchema(schema: object): Record<string, unknown> | null {
	try {
		return toAnthropicStrictToolSchema(schema);
	} catch (error) {
		if (error instanceof UnstrictifiableSchemaError) return null;
		throw error;
	}
}

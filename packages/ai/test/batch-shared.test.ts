import { describe, expect, it } from "vitest";
import {
	alignResults,
	batchDelay,
	buildResult,
	duplicateCustomIdFailures,
	failAll,
	failure,
	findDuplicateCustomIds,
	findInvalidCustomIds,
	resolveMaxPolls,
	resolvePollInterval,
	throwIfAborted,
} from "../src/api/batch/shared.ts";
import { type BatchItem, NotBatchableError } from "../src/types.ts";

const item = (customId: string, outputSchema?: Record<string, unknown>): BatchItem => ({
	customId,
	context: { messages: [{ role: "user", content: "hi", timestamp: 0 }] },
	...(outputSchema ? { outputSchema } : {}),
});

const SCHEMA = { type: "object", properties: { a: { type: "string" } }, required: ["a"] };

describe("duplicate customId guard", () => {
	it("finds duplicates", () => {
		expect(findDuplicateCustomIds([item("a"), item("b"), item("a")])).toEqual(["a"]);
	});

	it("returns null for a clean job", () => {
		expect(duplicateCustomIdFailures([item("a"), item("b")])).toBeNull();
	});

	it("fails EVERY item when any id is duplicated, because providers reject the whole job", () => {
		const results = duplicateCustomIdFailures([item("a"), item("b"), item("a")]);
		expect(results).not.toBeNull();
		expect(results).toHaveLength(3);
		expect(results?.every((r) => !r.ok)).toBe(true);
		// the offending id is attributed precisely...
		expect(results?.filter((r) => r.errorKind === "duplicate_custom_id").map((r) => r.customId)).toEqual(["a", "a"]);
		// ...and the innocent item is told why it did not run
		expect(results?.find((r) => r.customId === "b")?.errorKind).toBe("submit");
	});
});

describe("customId charset validation", () => {
	const ANTHROPIC = /^[a-zA-Z0-9_-]{1,64}$/;

	it("accepts conforming ids", () => {
		expect(findInvalidCustomIds([item("abc-123_XYZ")], ANTHROPIC)).toEqual([]);
	});

	it("rejects ids with disallowed characters or excess length", () => {
		expect(findInvalidCustomIds([item("has space"), item("a".repeat(65)), item("ok")], ANTHROPIC)).toEqual([
			"has space",
			"a".repeat(65),
		]);
	});
});

describe("buildResult", () => {
	it("returns raw text when no schema was declared", () => {
		const r = buildResult(item("a"), "plain text");
		expect(r).toMatchObject({ customId: "a", ok: true, text: "plain text" });
		expect(r.value).toBeUndefined();
	});

	it("parses against a declared schema", () => {
		const r = buildResult(item("a", SCHEMA), '{"a":"x"}');
		expect(r.ok).toBe(true);
		expect(r.value).toEqual({ a: "x" });
	});

	// Review 2026-07-26: buildResult used to only JSON.parse, so schema-invalid
	// output was returned as ok:true. That is reachable in production because
	// both Anthropic and OpenAI fall back to UNSTRICT generation when a schema
	// cannot be strictified. The previous version of this suite would have
	// passed with validation entirely absent — these cases pin that it isn't.
	it("rejects JSON that parses but violates the declared schema (wrong type)", () => {
		const r = buildResult(item("a", SCHEMA), '{"a":123}');
		expect(r.ok).toBe(false);
		expect(r.errorKind).toBe("parse");
		expect(r.error).toMatch(/did not satisfy the declared outputSchema/);
		expect(r.text).toBe('{"a":123}');
	});

	it("rejects JSON missing a required property", () => {
		const r = buildResult(item("a", SCHEMA), '{"b":"x"}');
		expect(r.ok).toBe(false);
		expect(r.errorKind).toBe("parse");
		expect(r.value).toBeUndefined();
	});

	it("reports unparseable output as a parse failure, keeping the raw text", () => {
		const r = buildResult(item("a", SCHEMA), "not json");
		expect(r.ok).toBe(false);
		expect(r.errorKind).toBe("parse");
		expect(r.text).toBe("not json");
	});

	it("reports a token-limit stop as `truncated`, NOT as a parse error", () => {
		// The JSON is invalid because generation ran out of room. Calling this a
		// parse error would point the caller at their schema instead of at
		// max_tokens, which is the actual cause.
		const r = buildResult(item("a", SCHEMA), '{"a":"xxx', undefined, true);
		expect(r.ok).toBe(false);
		expect(r.errorKind).toBe("truncated");
		expect(r.error).toContain("token limit");
	});
});

describe("alignResults", () => {
	it("preserves input order", () => {
		const items = [item("a"), item("b"), item("c")];
		const map = new Map(items.map((i) => [i.customId, { customId: i.customId, ok: true, text: i.customId }]));
		expect(alignResults(items, map).map((r) => r.customId)).toEqual(["a", "b", "c"]);
	});

	it("synthesises a failure for a customId the provider never returned", () => {
		const items = [item("a"), item("missing")];
		const map = new Map([["a", { customId: "a", ok: true, text: "x" }]]);
		const out = alignResults(items, map);
		expect(out).toHaveLength(2);
		expect(out[1]).toMatchObject({ customId: "missing", ok: false, errorKind: "provider_item" });
	});
});

describe("failAll", () => {
	it("marks every item failed with the same cause", () => {
		const out = failAll([item("a"), item("b")], "boom");
		expect(out.map((r) => r.customId)).toEqual(["a", "b"]);
		expect(out.every((r) => !r.ok && r.errorKind === "submit" && r.error === "boom")).toBe(true);
	});
});

describe("poll settings", () => {
	it("defaults to a 60s cadence and a 24h ceiling", () => {
		expect(resolvePollInterval()).toBe(60_000);
		expect(resolveMaxPolls()).toBe(1_440);
	});

	it("honours overrides and clamps to sane minimums", () => {
		expect(resolvePollInterval({ pollIntervalMs: 500 })).toBe(500);
		expect(resolvePollInterval({ pollIntervalMs: 0 })).toBe(1);
		expect(resolveMaxPolls({ maxPolls: 0 })).toBe(1);
	});
});

describe("abort handling", () => {
	it("throws immediately on an already-aborted signal", () => {
		const c = new AbortController();
		c.abort();
		expect(() => throwIfAborted(c.signal)).toThrow(/aborted/i);
	});

	it("rejects an in-flight delay when the signal fires", async () => {
		const c = new AbortController();
		const pending = batchDelay(10_000, c.signal);
		c.abort();
		await expect(pending).rejects.toThrow(/aborted/i);
	});

	it("resolves normally without a signal", async () => {
		await expect(batchDelay(1)).resolves.toBeUndefined();
	});
});

describe("NotBatchableError", () => {
	it("names the provider and model, and states that batch cannot be emulated", () => {
		const err = new NotBatchableError("google-vertex", "gemini-2.5-pro");
		expect(err.name).toBe("NotBatchableError");
		expect(err.provider).toBe("google-vertex");
		expect(err.model).toBe("gemini-2.5-pro");
		expect(err.message).toContain("NOT_BATCHABLE");
		expect(err.message).toContain("cannot be emulated");
	});
});

describe("failure()", () => {
	it("carries a machine-readable kind alongside the message", () => {
		expect(failure("a", "why", "provider_item")).toEqual({
			customId: "a",
			ok: false,
			error: "why",
			errorKind: "provider_item",
		});
	});
});

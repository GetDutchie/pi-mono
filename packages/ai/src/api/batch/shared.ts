/**
 * Shared batch-transport plumbing: correlation-id hygiene, poll/abort timing,
 * and result helpers. Provider-specific wire formats live in the sibling
 * transports.
 */

import type { BatchErrorKind, BatchItem, BatchOptions, BatchResult, Usage } from "../../types.ts";

export const DEFAULT_POLL_INTERVAL_MS = 60_000;
/** 24h at the default cadence. */
export const DEFAULT_MAX_POLLS = 1_440;

export function resolvePollInterval(options?: BatchOptions): number {
	return Math.max(1, options?.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS);
}

export function resolveMaxPolls(options?: BatchOptions): number {
	return Math.max(1, options?.maxPolls ?? DEFAULT_MAX_POLLS);
}

export class BatchAbortError extends Error {
	constructor() {
		super("Batch operation aborted");
		this.name = "BatchAbortError";
	}
}

export function throwIfAborted(signal?: AbortSignal): void {
	if (signal?.aborted) throw new BatchAbortError();
}

/** Abort-aware sleep. Rejects immediately when the signal fires. */
export function batchDelay(ms: number, signal?: AbortSignal): Promise<void> {
	throwIfAborted(signal);
	return new Promise((resolve, reject) => {
		const timer = setTimeout(() => {
			signal?.removeEventListener("abort", onAbort);
			resolve();
		}, ms);
		const onAbort = () => {
			clearTimeout(timer);
			reject(new BatchAbortError());
		};
		signal?.addEventListener("abort", onAbort, { once: true });
	});
}

// -----------------------------------------------------------------------------
// Correlation ids
// -----------------------------------------------------------------------------

/**
 * Duplicate-customId guard. Every provider batch surface rejects a job outright
 * when two requests share a correlation id, and that rejection kills every
 * OTHER request in the job too, not just the offending pair — with no per-item
 * attribution in the error. Detecting it locally turns an opaque whole-job
 * failure into precise per-item results and saves a guaranteed-doomed round
 * trip.
 */
export function findDuplicateCustomIds(items: readonly { customId: string }[]): string[] {
	const seen = new Set<string>();
	const dupes = new Set<string>();
	for (const item of items) {
		if (seen.has(item.customId)) dupes.add(item.customId);
		seen.add(item.customId);
	}
	return [...dupes];
}

/** Per-item failures for every duplicate-id item, or null when the job is clean. */
export function duplicateCustomIdFailures(items: readonly BatchItem[]): BatchResult[] | null {
	const dupes = new Set(findDuplicateCustomIds(items));
	if (dupes.size === 0) return null;
	return items.map((item) =>
		dupes.has(item.customId)
			? failure(
					item.customId,
					`duplicate customId "${item.customId}" in this job — providers reject the whole batch for this, so it was caught before submit`,
					"duplicate_custom_id",
				)
			: failure(item.customId, "not submitted: another item in this job had a duplicate customId", "submit"),
	);
}

/**
 * Validate ids against a provider's charset rule. Returns the offending ids.
 * The rule is provider-specific — Anthropic is the strictest — so transports
 * supply their own pattern rather than sharing a lowest-common-denominator one.
 */
export function findInvalidCustomIds(items: readonly { customId: string }[], pattern: RegExp): string[] {
	return items.filter((item) => !pattern.test(item.customId)).map((item) => item.customId);
}

// -----------------------------------------------------------------------------
// Results
// -----------------------------------------------------------------------------

export function failure(customId: string, error: string, errorKind: BatchErrorKind): BatchResult {
	return { customId, ok: false, error, errorKind };
}

/** Same failure for every item — submit-time faults are whole-job by nature. */
export function failAll(
	items: readonly BatchItem[],
	error: string,
	errorKind: BatchErrorKind = "submit",
): BatchResult[] {
	return items.map((item) => failure(item.customId, error, errorKind));
}

/**
 * Build a success result, parsing against the item's schema when it declared
 * one. A truncated generation is reported as `truncated` rather than `parse`:
 * the JSON is invalid because the model ran out of room, and telling the caller
 * "bad JSON" would point them at the schema instead of at max_tokens.
 */
export function buildResult(item: BatchItem, text: string, usage?: Usage, truncated = false): BatchResult {
	if (truncated) {
		return {
			...failure(
				item.customId,
				"generation stopped at the token limit; output is truncated" +
					(item.outputSchema ? " and structured output is therefore incomplete" : ""),
				"truncated",
			),
			text,
			usage,
		};
	}
	if (!item.outputSchema) return { customId: item.customId, ok: true, text, usage };
	try {
		return { customId: item.customId, ok: true, text, value: JSON.parse(text), usage };
	} catch (err) {
		return {
			...failure(item.customId, `output did not parse as JSON: ${(err as Error).message}`, "parse"),
			text,
			usage,
		};
	}
}

/**
 * Order results to match `items`, filling in a failure for any customId the
 * provider never reported back. A silently-missing item would otherwise look
 * like a short result array and be easy to miss.
 */
export function alignResults(items: readonly BatchItem[], byCustomId: Map<string, BatchResult>): BatchResult[] {
	return items.map(
		(item) =>
			byCustomId.get(item.customId) ??
			failure(item.customId, "provider returned no result for this customId", "provider_item"),
	);
}

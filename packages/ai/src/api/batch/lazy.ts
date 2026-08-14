import type { ProviderBatch } from "../../types.ts";

/**
 * Wraps a dynamically imported batch-transport module as `ProviderBatch`,
 * mirroring `lazyApi` for streams: the provider SDK is only imported when a
 * batch call is actually made, so declaring batch support costs nothing at
 * startup.
 *
 * Each transport gets its own `<name>.lazy.ts` module. Co-locating several
 * dynamic imports in one module would defeat tree-shaking: a bundler that
 * inlines dynamic imports pulls every transport (and its SDK) into any bundle
 * that touches one of them.
 */
export function lazyBatch(
	load: () => Promise<{ default?: ProviderBatch } & Record<string, unknown>>,
	key: string,
): ProviderBatch {
	return {
		async submitBatch(model, items, options) {
			return ((await load())[key] as ProviderBatch).submitBatch(model, items, options);
		},
		async pollBatch(model, batchId, items, options) {
			return ((await load())[key] as ProviderBatch).pollBatch(model, batchId, items, options);
		},
		async submitAndAwait(model, items, options, hooks) {
			return ((await load())[key] as ProviderBatch).submitAndAwait(model, items, options, hooks);
		},
	};
}

import type { ProviderBatch } from "../../types.ts";

/**
 * Lazy batch-transport wrappers, mirroring `api/*.lazy.ts` for streams: the
 * provider SDK is only imported when a batch call is actually made, so
 * declaring batch support costs nothing at startup.
 */
const lazyBatch = (
	load: () => Promise<{ default?: ProviderBatch } & Record<string, unknown>>,
	key: string,
): ProviderBatch => ({
	async submitBatch(model, items, options) {
		return ((await load())[key] as ProviderBatch).submitBatch(model, items, options);
	},
	async pollBatch(model, batchId, items, options) {
		return ((await load())[key] as ProviderBatch).pollBatch(model, batchId, items, options);
	},
	async submitAndAwait(model, items, options, hooks) {
		return ((await load())[key] as ProviderBatch).submitAndAwait(model, items, options, hooks);
	},
});

export const anthropicBatchApi = (): ProviderBatch => lazyBatch(() => import("./anthropic-batch.ts"), "anthropicBatch");
export const openaiBatchApi = (): ProviderBatch => lazyBatch(() => import("./openai-batch.ts"), "openaiBatch");
export const googleBatchApi = (): ProviderBatch => lazyBatch(() => import("./google-batch.ts"), "googleBatch");

import type { ProviderBatch } from "../../types.ts";
import { lazyBatch } from "./lazy.ts";

export const anthropicBatchApi = (): ProviderBatch => lazyBatch(() => import("./anthropic-batch.ts"), "anthropicBatch");

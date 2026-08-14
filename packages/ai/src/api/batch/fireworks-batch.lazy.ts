import type { ProviderBatch } from "../../types.ts";
import { lazyBatch } from "./lazy.ts";

export const fireworksBatchApi = (): ProviderBatch => lazyBatch(() => import("./fireworks-batch.ts"), "fireworksBatch");

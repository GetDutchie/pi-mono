import type { ProviderBatch } from "../../types.ts";
import { lazyBatch } from "./lazy.ts";

export const googleBatchApi = (): ProviderBatch => lazyBatch(() => import("./google-batch.ts"), "googleBatch");

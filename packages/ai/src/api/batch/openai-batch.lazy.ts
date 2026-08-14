import type { ProviderBatch } from "../../types.ts";
import { lazyBatch } from "./lazy.ts";

export const openaiBatchApi = (): ProviderBatch => lazyBatch(() => import("./openai-batch.ts"), "openaiBatch");

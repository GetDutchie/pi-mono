import { anthropicMessagesApi } from "../api/anthropic-messages.lazy.ts";
import { fireworksBatchApi } from "../api/batch/lazy.ts";
import { openAICompletionsApi } from "../api/openai-completions.lazy.ts";
import { envApiKeyAuth } from "../auth/helpers.ts";
import { createProvider, type Provider } from "../models.ts";
import { FIREWORKS_MODELS } from "./fireworks.models.ts";

export function fireworksProvider(): Provider<"anthropic-messages" | "openai-completions"> {
	return createProvider({
		id: "fireworks",
		name: "Fireworks",
		baseUrl: "https://api.fireworks.ai/inference",
		auth: { apiKey: envApiKeyAuth("Fireworks API key", ["FIREWORKS_API_KEY"]) },
		models: Object.values(FIREWORKS_MODELS),
		api: {
			"anthropic-messages": anthropicMessagesApi(),
			"openai-completions": openAICompletionsApi(),
		},
		// One transport for every model regardless of the realtime `api`: Fireworks
		// batch is its own account-scoped control plane, and a batch line's body is
		// Chat-Completions-shaped even for models served realtime over
		// anthropic-messages. Requires FIREWORKS_ACCOUNT_ID.
		batch: fireworksBatchApi(),
	});
}

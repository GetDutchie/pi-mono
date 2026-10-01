/**
 * Temporary compatibility entrypoint preserving the old global pi-ai API
 * surface: api-dispatch `stream()`/`complete()` with env API key injection,
 * the api-registry, generated catalog reads (`getModel`/`getModels`/
 * `getProviders`), per-API lazy stream wrappers, and image generation.
 *
 * Existing apps switch imports from "@earendil-works/pi-ai" to
 * "@earendil-works/pi-ai/compat" unchanged; new code uses `createModels()`
 * and the provider factories. This module is deleted with the coding-agent
 * ModelManager migration.
 */

export * from "./api/anthropic-messages.lazy.ts";
export * from "./api/azure-openai-responses.lazy.ts";
export * from "./api/bedrock-converse-stream.lazy.ts";
export * from "./api/google-generative-ai.lazy.ts";
export * from "./api/google-vertex.lazy.ts";
export * from "./api/mistral-conversations.lazy.ts";
export * from "./api/openai-codex-responses.lazy.ts";
export * from "./api/openai-completions.lazy.ts";
export * from "./api/openai-responses.lazy.ts";
export * from "./api/pi-messages.lazy.ts";
export * from "./env-api-keys.ts";
export * from "./image-models.ts";
export * from "./images.ts";
export * from "./images-api-registry.ts";
export * from "./index.ts";
export * from "./legacy-api-aliases.ts";
export * from "./providers/images/register-builtins.ts";

import type { Static, TSchema } from "typebox";
import { anthropicMessagesApi } from "./api/anthropic-messages.lazy.ts";
import { azureOpenAIResponsesApi } from "./api/azure-openai-responses.lazy.ts";
import { bedrockConverseStreamApi } from "./api/bedrock-converse-stream.lazy.ts";
import { resolveJsonSchemaStrictSampling } from "./api/constrained-sampling.ts";
import { googleGenerativeAIApi } from "./api/google-generative-ai.lazy.ts";
import { googleVertexApi } from "./api/google-vertex.lazy.ts";
import { mistralConversationsApi } from "./api/mistral-conversations.lazy.ts";
import { openAICodexResponsesApi } from "./api/openai-codex-responses.lazy.ts";
import { openAICompletionsApi } from "./api/openai-completions.lazy.ts";
import { openAIResponsesApi } from "./api/openai-responses.lazy.ts";
import { piMessagesApi } from "./api/pi-messages.lazy.ts";
import { getEnvApiKey } from "./env-api-keys.ts";
import type { ModelsApiStreamOptions } from "./models.ts";
import { builtinModels, getBuiltinModel, getBuiltinModels, getBuiltinProviders } from "./providers/all.ts";

export type { BuiltinProvider } from "./providers/all.ts";

import { createFauxCore, type FauxProviderRegistration, type RegisterFauxProviderOptions } from "./providers/faux.ts";
import type {
	Api,
	ApiStreamOptions,
	AssistantMessage,
	AssistantMessageEventStream,
	Context,
	JsonObject,
	Model,
	ProviderStreamOptions,
	ProviderStreams,
	SimpleStreamOptions,
	StreamFunction,
	StreamOptions,
	TextContent,
	Tool,
	ToolCall,
	TranscriptContext,
} from "./types.ts";
import { appendAssistantMessageDiagnostic } from "./utils/diagnostics.ts";
import { normalizeContext } from "./utils/transcript.ts";
import { validateToolArguments } from "./utils/validation.ts";

/** @deprecated Static catalog read. Use `getBuiltinModel` from "@earendil-works/pi-ai/providers/all" or `Models.getModel()`. */
export const getModel = getBuiltinModel;

/** @deprecated Static catalog read. Use `getBuiltinModels` from "@earendil-works/pi-ai/providers/all" or `Models.getModels()`. */
export const getModels = getBuiltinModels;

/** @deprecated Static catalog read. Use `getBuiltinProviders` from "@earendil-works/pi-ai/providers/all" or `Models.getProviders()`. */
export const getProviders = getBuiltinProviders;

export type ApiStreamFunction = (
	model: Model<Api>,
	context: TranscriptContext,
	options?: StreamOptions,
) => AssistantMessageEventStream;

export type ApiStreamSimpleFunction = (
	model: Model<Api>,
	context: TranscriptContext,
	options?: SimpleStreamOptions,
) => AssistantMessageEventStream;

export interface ApiProvider<TApi extends Api = Api, TOptions extends StreamOptions = StreamOptions> {
	api: TApi;
	stream: StreamFunction<TApi, TOptions>;
	streamSimple: StreamFunction<TApi, SimpleStreamOptions>;
}

interface ApiProviderInternal {
	api: Api;
	stream: ApiStreamFunction;
	streamSimple: ApiStreamSimpleFunction;
}

type RegisteredApiProvider = {
	provider: ApiProviderInternal;
	sourceId?: string;
};

const apiProviderRegistry = new Map<string, RegisteredApiProvider>();

function wrapStream<TApi extends Api, TOptions extends StreamOptions>(
	api: TApi,
	stream: StreamFunction<TApi, TOptions>,
): ApiStreamFunction {
	return (model, context, options) => {
		if (model.api !== api) {
			throw new Error(`Mismatched api: ${model.api} expected ${api}`);
		}
		return stream(model as Model<TApi>, context, options as TOptions);
	};
}

function wrapStreamSimple<TApi extends Api>(
	api: TApi,
	streamSimple: StreamFunction<TApi, SimpleStreamOptions>,
): ApiStreamSimpleFunction {
	return (model, context, options) => {
		if (model.api !== api) {
			throw new Error(`Mismatched api: ${model.api} expected ${api}`);
		}
		return streamSimple(model as Model<TApi>, context, options);
	};
}

export function registerApiProvider<TApi extends Api, TOptions extends StreamOptions>(
	provider: ApiProvider<TApi, TOptions>,
	sourceId?: string,
): void {
	apiProviderRegistry.set(provider.api, {
		provider: {
			api: provider.api,
			stream: wrapStream(provider.api, provider.stream),
			streamSimple: wrapStreamSimple(provider.api, provider.streamSimple),
		},
		sourceId,
	});
}

export function getApiProvider(api: Api): ApiProviderInternal | undefined {
	return apiProviderRegistry.get(api)?.provider;
}

export function getApiProviders(): ApiProviderInternal[] {
	return Array.from(apiProviderRegistry.values(), (entry) => entry.provider);
}

export function unregisterApiProviders(sourceId: string): void {
	for (const [api, entry] of apiProviderRegistry.entries()) {
		if (entry.sourceId === sourceId) {
			apiProviderRegistry.delete(api);
		}
	}
}

function clearApiProviders(): void {
	apiProviderRegistry.clear();
}

export function registerFauxProvider(options: RegisterFauxProviderOptions = {}): FauxProviderRegistration {
	const core = createFauxCore(options);
	const sourceId = `faux-provider-${Math.random().toString(36).slice(2, 10)}`;
	registerApiProvider({ api: core.api, stream: core.stream, streamSimple: core.streamSimple }, sourceId);
	return {
		api: core.api,
		models: core.models,
		getModel: core.getModel,
		state: core.state,
		setResponses: core.setResponses,
		appendResponses: core.appendResponses,
		getPendingResponseCount: core.getPendingResponseCount,
		unregister() {
			unregisterApiProviders(sourceId);
		},
	};
}

const BUILTIN_APIS: [Api, ProviderStreams][] = [
	["anthropic-messages", anthropicMessagesApi()],
	["openai-completions", openAICompletionsApi()],
	["openai-responses", openAIResponsesApi()],
	["openai-codex-responses", openAICodexResponsesApi()],
	["azure-openai-responses", azureOpenAIResponsesApi()],
	["google-generative-ai", googleGenerativeAIApi()],
	["google-vertex", googleVertexApi()],
	["mistral-conversations", mistralConversationsApi()],
	["bedrock-converse-stream", bedrockConverseStreamApi()],
	["pi-messages", piMessagesApi()],
];

const builtinApiProviderInstances = new Map<Api, ReturnType<typeof getApiProvider>>();

/**
 * Registers the builtin API implementations into the api-registry without
 * clobbering existing entries: compat may load after a test or extension has
 * already registered an override for a builtin api id.
 */
export function registerBuiltInApiProviders(): void {
	for (const [api, streams] of BUILTIN_APIS) {
		if (!getApiProvider(api)) {
			registerApiProvider({ api, stream: streams.stream, streamSimple: streams.streamSimple });
		}
		builtinApiProviderInstances.set(api, getApiProvider(api));
	}
}

export function resetApiProviders(): void {
	clearApiProviders();
	builtinApiProviderInstances.clear();
	registerBuiltInApiProviders();
}

registerBuiltInApiProviders();

const compatModels = builtinModels();
const AMBIENT_AUTH_MARKER = "<authenticated>";

function hasExplicitApiKey(apiKey: string | undefined): apiKey is string {
	return typeof apiKey === "string" && apiKey.trim().length > 0;
}

function withEnvApiKey<TOptions extends StreamOptions>(
	model: Model<Api>,
	options: TOptions | undefined,
): TOptions | undefined {
	if (hasExplicitApiKey(options?.apiKey)) return options;
	const apiKey = getEnvApiKey(model.provider, options?.env);
	if (!apiKey || apiKey === AMBIENT_AUTH_MARKER) return options;
	return { ...options, apiKey } as TOptions;
}

function hasResolvedCloudflareAuth(options: StreamOptions | undefined): boolean {
	return hasExplicitApiKey(options?.apiKey) || typeof options?.headers?.["cf-aig-authorization"] === "string";
}

function getBuiltinProviderForModel(model: Model<Api>) {
	if (getApiProvider(model.api) !== builtinApiProviderInstances.get(model.api)) return undefined;
	const provider = compatModels.getProvider(model.provider);
	return provider?.getModels().some((candidate) => candidate.api === model.api) ? provider : undefined;
}

function resolveApiProvider(api: Api) {
	const provider = getApiProvider(api);
	if (!provider) {
		throw new Error(`No API provider registered for api: ${api}`);
	}
	return provider;
}

export function stream<TApi extends Api>(
	model: Model<TApi>,
	context: Context,
	options?: ProviderStreamOptions,
): AssistantMessageEventStream {
	const transcript = normalizeContext(context);
	const builtinProvider = getBuiltinProviderForModel(model);
	if (builtinProvider) {
		if (model.provider.startsWith("cloudflare-") && !hasResolvedCloudflareAuth(options)) {
			return compatModels.stream(model, transcript, options as ModelsApiStreamOptions<TApi> | undefined);
		}
		return builtinProvider.stream(model, transcript, withEnvApiKey(model, options) as ApiStreamOptions<TApi>);
	}
	const provider = resolveApiProvider(model.api);
	return provider.stream(model, transcript, withEnvApiKey(model, options) as StreamOptions);
}

export async function complete<TApi extends Api>(
	model: Model<TApi>,
	context: Context,
	options?: ProviderStreamOptions,
): Promise<AssistantMessage> {
	const s = stream(model, context, options);
	return s.result();
}

export function streamSimple<TApi extends Api>(
	model: Model<TApi>,
	context: Context,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream {
	const transcript = normalizeContext(context);
	const builtinProvider = getBuiltinProviderForModel(model);
	if (builtinProvider) {
		if (model.provider.startsWith("cloudflare-") && !hasResolvedCloudflareAuth(options)) {
			return compatModels.streamSimple(model, transcript, options);
		}
		return builtinProvider.streamSimple(model, transcript, withEnvApiKey(model, options));
	}
	const provider = resolveApiProvider(model.api);
	return provider.streamSimple(model, transcript, withEnvApiKey(model, options));
}

export async function completeSimple<TApi extends Api>(
	model: Model<TApi>,
	context: Context,
	options?: SimpleStreamOptions,
): Promise<AssistantMessage> {
	const s = streamSimple(model, context, options);
	return s.result();
}

/** A TypeBox schema or a plain JSON Schema object, such as one derived from Zod. */
export type StructuredOutputSchema = TSchema | Record<string, unknown>;

/** Preserves TypeBox inference while keeping JSON Schema output explicitly unknown. */
export type StructuredOutputValue<TSchemaValue extends StructuredOutputSchema> = TSchemaValue extends TSchema
	? Static<TSchemaValue>
	: unknown;

export interface StructuredCompletionOptions extends ProviderStreamOptions {
	toolName?: string;
	toolDescription?: string;
}

export interface StructuredCompletion<T> {
	value: T;
	message: AssistantMessage;
}

const DEFAULT_STRUCTURED_OUTPUT_TOOL_NAME = "submit_structured_output";
const DEFAULT_STRUCTURED_OUTPUT_TOOL_DESCRIPTION = "Return the requested structured output using this function.";
const STRUCTURED_TOOL_NAME_PATTERN = /^[A-Za-z0-9_-]{1,64}$/;
const ANTHROPIC_UNSUPPORTED_SCHEMA_KEYWORDS = new Set([
	"minimum",
	"maximum",
	"exclusiveMinimum",
	"exclusiveMaximum",
	"multipleOf",
	"minLength",
	"maxLength",
	"pattern",
	"minItems",
	"maxItems",
	"uniqueItems",
	"minProperties",
	"maxProperties",
	"not",
]);

function collectUnsupportedAnthropicKeywords(node: unknown, path: string, found: string[]): void {
	if (Array.isArray(node)) {
		for (const [index, item] of node.entries()) collectUnsupportedAnthropicKeywords(item, `${path}[${index}]`, found);
		return;
	}
	if (node === null || typeof node !== "object") return;
	for (const [key, value] of Object.entries(node)) {
		const childPath = path ? `${path}.${key}` : key;
		if (ANTHROPIC_UNSUPPORTED_SCHEMA_KEYWORDS.has(key)) found.push(childPath);
		else collectUnsupportedAnthropicKeywords(value, childPath, found);
	}
}

function assertAnthropicNativeSchemaCompatible(schema: Record<string, unknown>): void {
	const offending: string[] = [];
	collectUnsupportedAnthropicKeywords(schema, "", offending);
	if (offending.length > 0) {
		throw new Error(
			"Anthropic native structured output rejects these JSON Schema keywords; " +
				"remove them from the transmitted schema and enforce the constraints in " +
				`application-level validation instead: ${offending.join(", ")}`,
		);
	}
}

function usesNativeStructuredOutput(model: Model<Api>): boolean {
	if (model.api === "anthropic-messages") return true;
	if (model.api !== "bedrock-converse-stream") return false;
	const identifiers = `${model.id} ${model.name}`.toLowerCase();
	return /claude-(?:sonnet|haiku|opus)-4-(?:5|6)(?:[^0-9]|$)/.test(identifiers);
}

function supportsStructuredToolStrictSampling(model: Model<Api>): boolean {
	switch (model.api) {
		case "anthropic-messages":
			return (model.compat as { supportsStrictTools?: boolean } | undefined)?.supportsStrictTools ?? false;
		case "bedrock-converse-stream":
			return (model.compat as { supportsStrictMode?: boolean } | undefined)?.supportsStrictMode ?? false;
		case "google-generative-ai":
		case "google-vertex": {
			const match = model.id.toLowerCase().match(/^gemini(?:-live)?-(\d+)/);
			return match !== null && Number.parseInt(match[1], 10) >= 3;
		}
		case "mistral-conversations":
			return true;
		case "openai-completions":
		case "openai-responses":
		case "openai-codex-responses":
		case "azure-openai-responses":
			return (model.compat as { supportsStrictMode?: boolean } | undefined)?.supportsStrictMode !== false;
		default:
			return false;
	}
}

function structuredToolChoice(model: Model<Api>, toolName: string): Record<string, unknown> {
	switch (model.api) {
		case "anthropic-messages":
			return { toolChoice: { type: "tool", name: toolName, disableParallelToolUse: true } };
		case "openai-completions":
		case "mistral-conversations":
			return { toolChoice: { type: "function", function: { name: toolName } } };
		case "openai-responses":
		case "azure-openai-responses":
		case "openai-codex-responses":
			return { toolChoice: { type: "function", name: toolName } };
		case "google-generative-ai":
		case "google-vertex":
			return { toolChoice: "any" };
		default:
			return {};
	}
}

/** Produce one typed JSON value through native response formatting or a forced private tool call. */
export async function completeStructured<TParameters extends StructuredOutputSchema>(
	model: Model<Api>,
	context: Context,
	parameters: TParameters,
	options: StructuredCompletionOptions = {},
): Promise<StructuredCompletion<StructuredOutputValue<TParameters>>> {
	if (context.tools?.length) {
		throw new Error("completeStructured does not accept context.tools; structured output owns the only tool slot");
	}
	const toolName = options.toolName ?? DEFAULT_STRUCTURED_OUTPUT_TOOL_NAME;
	if (!STRUCTURED_TOOL_NAME_PATTERN.test(toolName))
		throw new Error(`Invalid structured output tool name: ${toolName}`);
	const toolDescription = options.toolDescription ?? DEFAULT_STRUCTURED_OUTPUT_TOOL_DESCRIPTION;
	if (!toolDescription.trim()) throw new Error("Structured output tool description must not be empty");

	const nativeStructured = usesNativeStructuredOutput(model);
	if (model.api === "bedrock-converse-stream" && !nativeStructured) {
		throw new Error(
			`Native Bedrock structured output is unsupported for ${model.id}. Use a Bedrock Claude 4.5/4.6 model or a provider with native schema-constrained output.`,
		);
	}
	if (model.api === "anthropic-messages") assertAnthropicNativeSchemaCompatible(parameters as Record<string, unknown>);
	const outputTool: Tool = {
		name: toolName,
		description: toolDescription,
		parameters: parameters as TSchema,
		constrainedSampling: { type: "json_schema", strict: "prefer" },
	};
	const { toolName: _toolName, toolDescription: _toolDescription, ...providerOptions } = options;
	const message = await complete(
		model,
		nativeStructured ? context : { ...context, tools: [outputTool] },
		nativeStructured
			? { ...providerOptions, outputSchema: { name: toolName, schema: parameters } }
			: { ...providerOptions, ...structuredToolChoice(model, toolName) },
	);
	if (
		!nativeStructured &&
		resolveJsonSchemaStrictSampling(
			outputTool,
			supportsStructuredToolStrictSampling(model),
			model.api === "anthropic-messages" ? (key) => ANTHROPIC_UNSUPPORTED_SCHEMA_KEYWORDS.has(key) : undefined,
		) !== true
	) {
		appendAssistantMessageDiagnostic(message, {
			type: "structured_output_strict_sampling_fallback",
			timestamp: Date.now(),
			details: { api: model.api, provider: model.provider, model: model.id, tool: toolName },
		});
	}
	if (message.stopReason === "error" || message.stopReason === "aborted") {
		throw new Error(message.errorMessage ?? `Structured completion failed with stop reason: ${message.stopReason}`);
	}

	if (nativeStructured) {
		const text = message.content
			.filter((block): block is TextContent => block.type === "text")
			.map((block) => block.text)
			.join("");
		let arguments_: unknown;
		try {
			arguments_ = JSON.parse(text);
		} catch {
			throw new Error("Native structured output was not valid JSON");
		}
		if (arguments_ === null || typeof arguments_ !== "object" || Array.isArray(arguments_)) {
			throw new Error("Native structured output must be a JSON object");
		}
		return {
			value: validateToolArguments(outputTool, {
				type: "toolCall",
				id: "native_structured_output",
				name: toolName,
				arguments: arguments_ as JsonObject,
			}) as StructuredOutputValue<TParameters>,
			message,
		};
	}

	const outputCalls = message.content.filter(
		(block): block is ToolCall => block.type === "toolCall" && block.name === toolName,
	);
	if (outputCalls.length !== 1) {
		throw new Error(`Structured completion expected exactly one ${toolName} call, received ${outputCalls.length}`);
	}
	return {
		value: validateToolArguments(outputTool, outputCalls[0]) as StructuredOutputValue<TParameters>,
		message,
	};
}

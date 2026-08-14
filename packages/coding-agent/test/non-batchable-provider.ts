import { type Api, type Model, NotBatchableError, type Provider } from "@earendil-works/pi-ai";

/**
 * The batch members every hand-rolled `Provider` literal must supply.
 *
 * `Provider` requires batch so that a provider built without `createProvider`
 * cannot silently drop the capability. Test stubs have no batch transport, so
 * they refuse loudly, exactly like `radius.ts` and the llama extension.
 */
export function nonBatchableProvider(
	providerId: string,
): Pick<Provider, "submitBatch" | "pollBatch" | "submitAndAwaitBatch" | "canBatch"> {
	const refuse = (model: Model<Api>) => Promise.reject(new NotBatchableError(providerId, model.id));
	return {
		canBatch: () => false,
		submitBatch: refuse,
		pollBatch: refuse,
		submitAndAwaitBatch: refuse,
	};
}

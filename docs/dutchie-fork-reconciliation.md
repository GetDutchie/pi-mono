# Dutchie fork reconciliation against upstream 0.82.1

**Written:** 2026-07-26. **Fork base:** `main` @ `76d300de` (2026-07-15).
**Upstream:** `earendil-works/pi-mono` @ `5bc1c2c0a` (2026-07-25), version **0.82.1**.
**Merge base:** `8479bd84`. **Behind:** 195 commits. **Ahead:** 29 (21 non-merge).

This document records the triage of the Dutchie patch line and the design rules
agreed for the batch work. It exists because the triage is expensive to redo —
it required diffing every Dutchie commit against upstream's new
`constrained-sampling` implementation and checking every downstream consumer.

---

## 1. Why this is a reconciliation, not a fast-forward

Upstream landed `24bace27c feat(ai): support constrained sampling (#6341)`
(2026-07-23) — a parallel implementation of the problem the Dutchie fork was
created to solve. It adds `packages/ai/src/api/constrained-sampling.ts` plus a
229-line test suite and touches **every** provider:

`anthropic-messages` · `azure-openai-responses` · `bedrock-converse-stream` ·
`google-generative-ai` · `google-shared` · `google-vertex` ·
`mistral-conversations` · `openai-codex-responses` · `openai-completions` ·
`openai-responses{,-shared}` · `types.ts`

New API — per-tool, richer than ours:

```ts
constrainedSampling?: false
  | { type: "json_schema"; strict: "prefer" | "require" }
  | { type: "grammar";     variants: GrammarVariants }
```

Upstream did **not** adopt our `utils/strict-tool-schema.ts` (it does not exist
on `upstream/main`). The two implementations collide in the same files, so this
is a hand-resolved merge.

### The two features are NOT the same thing

Repeatedly confused during triage — keep them distinct:

| | Scope | Provider coverage | Owner |
| --- | --- | --- | --- |
| `Tool.constrainedSampling` (upstream) | tool-call **arguments** | all providers | upstream |
| `StreamOptions.outputSchema` → `output_config.format` (Dutchie) | the **whole response** | Anthropic + Bedrock | Dutchie only — upstream has nothing |

Upstream's `output_config` only ever carries `effort`. There is no upstream
equivalent of response-level structured output.

---

## 2. Commit triage

### DROP — superseded by `#6341`

Tool-call strictification. Upstream's covers more providers with a better API;
keeping ours means maintaining a second implementation of the same predicate.

| Commit | Subject |
| --- | --- |
| `69e057100` | strictTools flag — native constrained decoding for tool-call arguments |
| `1d338674a` | thread strictTools through Agent and createAgentSession |
| `377a5cb58` | revert outputSchema/strictTools API surface |
| `21a51d08b` | replace ensure-json reprompt loop with provider-native structured output |
| `3d8f4ef35` | cross-vendor review of strict tool schemas (round 2) |
| `b88d80055` | allOf is unstrictifiable in the OpenAI strict subset (round 3) |
| `4185fbc48` | reject Anthropic-unsupported schema keywords |
| `00d1060c2` / `57f116f8e` | make provider tool-schema strictification opt-in per tool (#21) |

### KEEP — response-level structured output (live consumer)

| Commit | Subject |
| --- | --- |
| `acb3e6595` | native structured output via `StreamOptions.outputSchema` |
| `a9befc10b` | schema-constrained structured completion |
| `6b9f0f980` | use native Anthropic structured output |
| `90a6cf4aa` | use native Bedrock structured output |
| `a98fad6a5` | prevent parallel structured output tools |
| `057cbde2a` | merge adaptive effort into `output_config` instead of replacing it |
| `63a5c2004` | strictify the schema sent to Anthropic native structured output |

**Consumer proof.** `GetDutchie/alchemy` `origin/develop` (its real trunk; `main`
is the stable-promotion target) pins
`"@earendil-works/pi-ai": "npm:@getdutchie/pi-ai@0.80.6-dutchie.12"` and hit the
same bugs independently. From `tools/pipeline/lib/llm.ts:426`:

> Anthropic silently falls back to ADVISORY (unconstrained) decoding for such a
> schema — which lets a rare degenerate generation emit structurally-invalid
> JSON (**observed live: a `reasoning` field that ran away into thousands of
> commas until it hit max_tokens**).

That is the same failure `63a5c2004` fixes. The same file records `057cbde2a`
as `FIXED UPSTREAM (2026-07-15)`, letting them delete a thinking-OFF retry
band-aid. This work is load-bearing, not speculative.

### KEEP — genuinely ours, upstream still unfixed

| Commit | Subject | Verified |
| --- | --- | --- |
| `499d63666` | add CJS require condition to every exports subpath | upstream `package.json` exports still have only `types` + `import` |
| `ab182b3d5` / `4a7d6da00` | export compat for CommonJS resolution | same |
| `bb18ddf8a` | fail closed on non-OK / non-SSE Anthropic streaming + Claude-5 Bedrock cache gate | no upstream equivalent found |

Both are plain upstream bugs, not Dutchie-specific. **Good upstream PR
candidates** — if they land, the fork shrinks to the structured-output line.

### KEEP (meta)

`6ae3e7b1b`, `76d300de8` — dutchie cut publish-flow docs.

---

## 3. Batch support — BUILT (`2885192b1`)

> **Status:** implemented on this branch. `ProviderBatch` ships with three
> transports; the analysis below is retained because it is the rationale.
> Re-verified after the merge: upstream 0.82.1 added **no** batch support
> (zero batch commits in the 195, one incidental occurrence of the word in
> `packages/ai/src`).

`grep -c batch` across all of pi-ai `dist`: **1**. There is no batch API.

Consequently `alchemy/tools/pipeline/lib/llm-batch.ts` (1,517 lines) bypasses
pi-ai entirely and hand-builds provider batch requests, reimplementing the
schema transform as `toConstrainedOutputSchema`. **That makes three copies of
the same transform** — fork (`anthropicStrictToolSchema`), upstream
(`constrained-sampling.ts`), alchemy (`toConstrainedOutputSchema`). It also
hand-maintains a regex→price table that duplicates pi-ai's generated
`ModelCost` catalog.

The file is remarkably portable: its only imports are `node:crypto`, `zod`, and
`./llm`. No db, no temporal, no fs.

### Shipped surface

pi-ai already had the seam — `ProviderStreams { stream, streamSimple }` and
`ProviderImages { generateImages }` are optional per-provider capability
modules. Batch slots in the same way, resolved identically (single
implementation, or a map keyed by `model.api` for mixed-API providers):

```ts
export interface ProviderBatch {
  submitBatch(model, items, options?): Promise<{ batchId: string }>;
  pollBatch(model, batchId, items, options?): Promise<BatchResult[]>;
  submitAndAwait(model, items, options?, hooks?): Promise<BatchResult[]>;
}
```

Surfaced on `Provider` as `submitBatch` / `pollBatch` / `submitAndAwaitBatch`,
plus `canBatch(model)` as the capability probe.

A `BatchItem` carries a full `Context` rather than a flattened
prompt/system pair, so a batch item is the same shape as a realtime call and
work moves between the two without being rewritten.

| Transport | File | Wired to |
| --- | --- | --- |
| Anthropic Message Batches | `api/batch/anthropic-batch.ts` | `anthropic` |
| Gemini Batch Mode (inline) | `api/batch/google-batch.ts` | `google` |
| OpenAI `/v1/files` + `/v1/batches` | `api/batch/openai-batch.ts` | **nothing — see below** |

Loudly not batchable: `openai`, `azure-openai-responses`, `fireworks`,
`google-vertex`, `mistral`, `bedrock`, `radius`.

### Why openai / azure / fireworks are NOT wired

They were, briefly. Adversarial review (gpt-5.6-sol, 2026-07-26) found all
three would return `canBatch() === true` and then fail on their first real
call:

- **azure-openai-responses** — generated models carry `baseUrl: ""`. The
  transport treated the empty string as falsy and omitted `baseURL`, so the
  OpenAI SDK would fall back to `https://api.openai.com/v1` and present an
  **Azure key to OpenAI**. Azure also needs resource/api-version resolution and
  `AZURE_OPENAI_BASE_URL` from `BatchOptions.env`.
- **openai** — every model in that catalog is `api: "openai-responses"`, but
  the transport emits `/v1/chat/completions` JSONL and decodes
  `choices[0].message.content`.
- **fireworks** — exposes neither OpenAI-style `/files` + `/batches` nor
  Anthropic `/v1/messages/batches`. Its batch inference is a different API
  (datasets + `/v1/accounts/{account}/batchInferenceJobs`). **API-compat does
  not imply batch-compat** — that was an unverified assumption.

The transport is kept, unwired, with those traps documented at the top of the
file. Two verified transports beat three where two are wrong. Note this means
the cheap Fireworks GLM/Kimi/MiniMax seats are **not** batchable yet.

**Gemini specifics worth remembering.** Inline requests have no `key` field —
that belongs to the JSONL file format — so the correlation id rides
`InlinedRequest.metadata`, which round-trips to `InlinedResponse.metadata`.
And the config key is `responseJsonSchema` (standard JSON Schema); the older
`responseSchema` expects Google's OpenAPI-ish dialect (uppercase
`"type": "OBJECT"`) and batch preprocessing rejects it with an opaque
`400 invalid argument`. Vertex uses the opposite dialect — one more reason it
is a separate transport rather than a flag.

### Provider knowledge encoded once, not per caller

- **Duplicate `customId`** is caught locally before submit. Providers reject the
  ENTIRE job over it and kill every healthy request with no per-item
  attribution; catching it locally turns an opaque whole-job 400 into precise
  per-item results and saves a guaranteed-doomed round trip.
- **Token-limit stops** are reported as `truncated`, not as a JSON parse error.
  The JSON is invalid because generation ran out of room; calling it a parse
  error points the caller at their schema instead of at `max_tokens`.
- **A `customId` the provider never returns** becomes an explicit per-item
  failure rather than a silently short result array.
- **Rich `Context` is rejected, not flattened.** v1 serializes plain string
  conversations only. The first cut coerced with
  `typeof content === "string" ? content : ""`, which turned multimodal parts,
  tool calls and thinking blocks into empty messages and dropped
  `Context.tools` — submitting a materially different prompt while returning a
  normal batch id. Same class of dishonesty as emulating batch.
- **Output is validated against `outputSchema`, not merely parsed.** Providers
  fall back to unstrict generation when a schema cannot be compiled to a
  grammar, so schema-invalid JSON genuinely reaches the result path.

### Ported / dropped (as built)

| Ported | Dropped |
| --- | --- |
| Anthropic Message Batches transport | `RealtimeEmulatingBatchTransport` — see §4 |
| OpenAI-compatible `/files` + `/batches` transport | `PRICES_PER_MTOK` / `lookupPricePerMTok` / `estimateCostUsd` — pi-ai's generated `ModelCost` replaces it |
| `customId` validation + duplicate guard | `PF_REALTIME_EMULATION_CONCURRENCY`, `PF_AZURE_OPENAI_BATCH_ENABLED` |
| JSONL result mappers | |
| Usage extraction, poll/backoff/abort plumbing | |

Rewritten rather than copied: the transports use the provider SDKs already
depended on (`@anthropic-ai/sdk`, `openai`, `@google/genai`) instead of raw
`fetch`, matching how the streaming paths work, which is why the result is
substantially smaller than the 1,517 lines it replaces.

### Still open

- **No live smoke test.** Both wired transports are typechecked and unit-tested
  against the SDK shapes; neither has submitted a real job.
- **No mocked-SDK transport tests** capturing actual request bodies or mapping
  representative provider responses (sol MINOR-1). Owed for Anthropic and
  Gemini.
- **Per-request abort plumbing.** `throwIfAborted` guards submit and the poll
  loop, but the SDK-level signals (`RequestOptions.signal`,
  `CreateBatchJobConfig.abortSignal`) are not passed into retrieve/download.
- **Gemini file mode (>20MB)** throws rather than silently splitting.
- **Fireworks batch** needs its real dataset/job transport before those seats
  can batch.
- **`packages/coding-agent` is untouched** — batch is pi-ai-only so far.

---

## 4. Design rule: silent degradation is a defect

**Operator ruling (Jordan, 2026-07-26).** Batch is a *pricing* decision (~50%
off). Silently emulating it with realtime calls means the caller pays full
price while believing they are on batch rates — invisible, because the results
are shape-identical.

Alchemy's `RealtimeEmulatingBatchTransport` is the anti-pattern, and it is worse
than it looks: it mints a **synthetic batch id**
(`realtime-${Date.now()}-...`) and writes it to the `batch_submitted` ledger
row, so the audit trail records a batch that never happened. Its own docstring
concedes "no 50%-off batch pricing, and per-item usage is not reported" — so
cost tracking, the one signal that would reveal it, also goes blind. And
`resolveBatchTransport` makes it the **default** for `azure-openai` unless
`PF_AZURE_OPENAI_BATCH_ENABLED=1`: opt in to cheap, silently default to
expensive.

### Rules for `ProviderBatch`

1. `batch` is genuinely optional on a provider. Absent means absent.
2. Requesting batch on a provider without it throws a typed `NotBatchableError`
   naming provider + model. **Never emulate, never degrade.**
3. **No synthetic batch ids, ever.** A batch id means a provider-side job exists.
4. Realtime is the caller's explicit choice at the call site, not a library
   default. Alchemy's existing error already prescribes `transport: "realtime"`,
   which makes the emulator redundant with a path they already have.
5. Batch API present but unreachable (creds/SKU) is also loud — "batch
   unavailable", never "batch emulated".

Do port `AzureBatchNotConfiguredTransport`'s behaviour: per-item `ok: false`
with a `NOT_BATCHABLE:` error naming the explicit opt-out.

### Corollary: strict is loud but NOT fatal

When `anthropicStrictToolSchema` / `strictToolSchema` returns `null`, the tool
is currently sent non-strict and silently falls back to the reprompt loop.

Unlike batch, strict **has a backstop** — local validate-and-retry still
enforces the schema, so correctness is preserved and only cost/latency degrade.
So: keep the fallback, but **emit a diagnostic** when a declared schema fails to
strictify. A schema that quietly stops strictifying after a refactor is a real
regression nobody would otherwise notice.

- batch missing → **fatal**
- strict unavailable → **loud, non-fatal**

---

## 5. Sequence

1. Merge `upstream/main` (0.82.1) into `main`; resolve constrained-sampling
   files in favour of upstream; drop the §2 DROP commits' effects.
2. Re-apply / verify the §2 KEEP set on top.
3. Add `ProviderBatch` per §3–4.
4. Cut `0.82.1-dutchie.1` for `pi-ai` and `pi-coding-agent` (new base version
   resets the suffix counter).
5. Bump consumers: URA and alchemy `develop`, both currently on
   `0.80.6-dutchie.12`.
6. Offer upstream as separate PRs: CJS exports, fail-closed streaming, and
   `ProviderBatch` (net-new capability; does not compete with
   `constrained-sampling` — job-level vs tool-level).

### Consumer impact

- **URA** — uses the tool path (`emit_result`). Its in-flight branch
  `fix/anthropic-strict-tools` sets `tool.strict = true`; under the new API that
  becomes `constrainedSampling: { type: "json_schema", strict: "require" }`.
  That branch needs a follow-up commit, or should be held until this lands.
- **alchemy `develop`** — uses the response path plus its own batch layer.
  Once `ProviderBatch` ships it deletes ~1,230 lines, its private
  `toConstrainedOutputSchema`, and its price table.

### Housekeeping

`git worktree prune` — several `/tmp/pi-mono-release-dutchie-*` worktrees from
cuts 6–9 are stale.

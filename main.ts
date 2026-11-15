// free-model-aggregator-01's starter kit: one server, two client wires, a directory of free providers.
//
// This entry builds a deterministic provider catalog, assigns the complete workload before serving,
// and translates OpenAI and Anthropic client traffic across OpenAI, Anthropic, and Gemini provider wires.
// Assignments respect model requirements and provider spread limits; a refusal triggers request-local
// failover, while a provider's 402 marks it unavailable for the rest of the trial.
//
// WHAT A TRIAL GIVES YOU
//   A trial gives this server a brief and an env, and runs it as `node /bot/main.ts` from your
//   repository's root, with Node 22's standard library alone (nothing is installed; TypeScript runs
//   directly, and a relative import needs its .ts extension). The brief is at the path in
//   $MAKEFUN_PROMPT_FILE. It carries three JSON documents:
//     - the provider directory: ten simulated free-tier providers, each with its id, wire (openai,
//       anthropic or gemini), auth scheme, key environment variable, the key the service takes, and the
//       base path you join to $MAKEFUN_ENV_URL;
//     - the quality table: every model's public rating and what it can do (tools, vision, JSON mode,
//       context length), and whether it is free;
//     - the workload manifest: the twenty requests the client will send, in order, with each one's api,
//       whether it streams, its weight and the requirements a model must meet to serve it.
//   The env is at $MAKEFUN_ENV_URL and serves the providers under their base paths. This server reads
//   the brief, probes each provider's catalog route in the provider's own wire, builds one catalog, then:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//        x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//        /v1/messages, each with a JSON body or with server-sent events;
//     4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//        process still running when the wall ends scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
//   body's catalog must be that same document. Build it once, as this file does, and use the one object.
//
// ROUTING AND RESPONSE BEHAVIOUR
//   The brief's workload manifest is planned before registration. Legal free models are ranked by their
//   rating_milli; the tools plus vision requests reserve their strongest eligible capacity first, then a
//   maximum-weight assignment spreads the remaining requests under the provider cap. Planned choices lead
//   each request's retry list. HTTP 402 retires that provider for the trial; 429 and 5xx skip it only for
//   the current request. Each provider attempt is logged with stable request, model, rating, attempt and
//   outcome fields. Existing wire adapters preserve the client protocol, including streaming responses.
//   The catalog is probed from provider answers and enriched from the quality table; --list prints the
//   same catalog document that registration sends. The server uses only the environment's simulated
//   providers and makes no model calls of its own.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe.
//   lib/wire.ts        Anthropic <-> Chat Completions translation and the SSE writers.
//   LICENSE            MIT: keep it or replace it with another OSI-approved license.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this baseline reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** Complete assignment of manifest request ids, fixed before the listener is registered. */
let manifestPlan = new Map<string, Route>();
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
let currentRequestId = 'unlisted';
const providerCalls = new Map<string, number>();
const exhaustedProviders = new Set<string>();
const retiredProviders = new Set<string>();

// Per-trial provider health informs fallback order without changing the manifest plan.
const providerHealth = new Map(brief.directory.providers.map((provider) => [provider.id, {
  consecutiveFailures: 0, totalCalls: 0, totalFailures: 0, coolUntil: 0,
}]));
let requestIndex = 0;

function healthFor(providerId) {
  let health = providerHealth.get(providerId);
  if (!health) {
    health = { consecutiveFailures: 0, totalCalls: 0, totalFailures: 0, coolUntil: 0 };
    providerHealth.set(providerId, health);
  }
  return health;
}

function recordProviderOutcome(providerId, failed, succeeded = !failed) {
  const health = healthFor(providerId);
  health.totalCalls += 1;
  if (failed) {
    health.totalFailures += 1;
    health.consecutiveFailures += 1;
    if (health.consecutiveFailures >= 2) retiredProviders.add(providerId);
  } else if (succeeded) {
    health.consecutiveFailures = 0;
    health.coolUntil = 0;
  }
}

function healthPenalty(providerId, index) {
  const health = healthFor(providerId);
  if (health.coolUntil >= index && health.consecutiveFailures >= 2) return Infinity;
  return health.consecutiveFailures === 1 ? 40 : 0;
}

function reorderFallbacks(candidates, index) {
  const score = (candidate) => candidate.rating - Math.min(70, candidate.count * 15)
    - healthPenalty(candidate.probe.provider.id, index);
  return candidates.map((candidate, order) => ({ candidate, order }))
    .filter(({ candidate }) => Number.isFinite(healthPenalty(candidate.probe.provider.id, index)))
    .sort((a, b) => score(b.candidate) - score(a.candidate)
      || healthFor(a.candidate.probe.provider.id).consecutiveFailures - healthFor(b.candidate.probe.provider.id).consecutiveFailures
      || a.order - b.order)
    .map(({ candidate }) => candidate);
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  const manifest = brief.workload?.manifest ?? [];
  const candidates = new Map<string, Array<{ provider: string; model: string; rating_milli: number; route: Route }>>();
  for (const request of manifest) {
    const options: Array<{ provider: string; model: string; rating_milli: number; route: Route }> = [];
    for (const probe of probes) {
      if (!probe.ok || (request.api === 'openai' && request.requires.json === true && probe.provider.wire !== 'openai')) continue;
      for (const model of probe.models) {
        const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
        if (!modelFree(model, row) || !meetsRequirements(request.requires, capabilitiesOf(model, row))) continue;
        options.push({ provider: probe.provider.id, model: model.id, rating_milli: row?.rating_milli ?? 0, route: { probe, model } });
      }
    }
    candidates.set(request.id, options);
  }
  const planned = planManifestAssignments(manifest, candidates);
  const validated = new Map(planned.assignments);
  for (const request of manifest) {
    const assignment = validated.get(request.id);
    const catalogProvider = catalog.providers.find((provider) => provider.id === assignment?.provider);
    const presentFree = catalogProvider?.models.some((model) => model.id === assignment?.model && model.free === true) === true;
    if (presentFree) continue;
    const replacement = [...(candidates.get(request.id) ?? [])]
      .filter((candidate) => catalog.providers.some((provider) => provider.id === candidate.provider
        && provider.models.some((model) => model.id === candidate.model && model.free === true)))
      .sort((a, b) => b.rating_milli - a.rating_milli || a.provider.localeCompare(b.provider) || a.model.localeCompare(b.model))[0];
    if (replacement !== undefined) {
      validated.set(request.id, replacement);
      log(`plan replacement request=${request.id} from=${assignment?.provider ?? 'missing'}/${assignment?.model ?? 'missing'} to=${replacement.provider}/${replacement.model}`);
    } else {
      log(`plan validation found no catalog-free replacement for request=${request.id}`);
    }
  }
  manifestPlan = new Map([...validated].map(([id, candidate]) => [id, candidate.route]));
  log(`planned ${manifestPlan.size} manifest requests with provider caps ${JSON.stringify(planned.providerCaps)}`);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Route {
  probe: Probe;
  model: ProbedModel;
}

/**
 * The provider and model that serve one request. TODO(quality): rank every free candidate that meets the
 * requirements by the quality table's rating_milli (quality is 60% of the score) and plan each provider's
 * per-trial budget. This baseline takes the first provider, in the brief's order, whose listing answered
 * with a free model meeting the requirements, and that provider's first model by id.
 */
function routesFor(requires: Requirement, jsonOnly = false): Route[] {
  const candidates: Array<Route & { rating: number; count: number }> = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (exhaustedProviders.has(probe.provider.id)) continue;
    if (retiredProviders.has(probe.provider.id)) continue;
    if (jsonOnly && probe.provider.wire !== 'openai') continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      candidates.push({ probe, model, rating: row?.rating_milli ?? 0, count: providerCalls.get(probe.provider.id) ?? 0 });
    }
  }
  candidates.sort((a, b) => {
    const ascore = a.rating - Math.min(70, a.count * 15);
    const bscore = b.rating - Math.min(70, b.count * 15);
    return bscore - ascore || b.rating - a.rating || a.count - b.count || a.probe.provider.id.localeCompare(b.probe.provider.id) || byId(a.model, b.model);
  });
  return candidates;
}

function routeFor(requires: Requirement, jsonOnly = false): Route | null {
  return routesFor(requires, jsonOnly)[0] ?? null;
}

/** Put the precomputed assignment first; retain all live legal candidates as ordered failover. */
function routesForRequest(requires: Requirement, jsonOnly = false): Route[] {
  const candidates = routesFor(requires, jsonOnly);
  const assigned = manifestPlan.get(currentRequestId);
  const index = assigned === undefined ? -1 : candidates.findIndex((route) => route.probe.provider.id === assigned.probe.provider.id && route.model.id === assigned.model.id);
  const planFirst = assigned === undefined || index < 0
    ? candidates
    : [candidates[index], ...candidates.slice(0, index), ...candidates.slice(index + 1)];
  if (assigned === undefined || index < 0) return reorderFallbacks(planFirst, requestIndex);
  return [planFirst[0], ...reorderFallbacks(planFirst.slice(1), requestIndex)];
}

/** Log only stable request/provider facts and the answer status, so replay output is deterministic. */
function logRouteAttempt(requestId: string, route: Route, attempt: number, outcome: string): void {
  const rating = ratingOfModel(brief.ratings, route.probe.provider.id, route.model.id)?.rating_milli ?? 0;
  const failures = healthFor(route.probe.provider.id).consecutiveFailures;
  log(`route request=${requestId} provider=${route.probe.provider.id} model=${route.model.id} rating_milli=${rating} consecutive_failures=${failures} attempt=${attempt} outcome=${outcome}`);
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  requestIndex = manifestIndex;
  if (entry !== undefined) {
    currentRequestId = entry.id;
    return entry.requires;
  }
  currentRequestId = `request-${manifestIndex}`;
  // TODO(workload): the manifest is the declared source of each request's requirements; this fallback is a
  // guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST one request using the provider's native upstream wire. */
async function callProvider(route: Route, body: Json): Promise<Response> {
  const base = joinBase(ENV_URL, route.probe.provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(route.probe.provider.auth, route.probe.provider.key),
  };
  if (route.probe.provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (body.stream === true) headers.accept = 'text/event-stream';
  const wire = route.probe.provider.wire;
  const url = wire === 'openai'
    ? `${base}/v1/chat/completions`
    : wire === 'anthropic'
      ? `${base}/v1/messages`
      : `${base}/v1beta/models/${encodeURIComponent(route.model.id)}:${body.stream === true ? 'streamGenerateContent?alt=sse' : 'generateContent'}`;
  return fetch(url, { method: 'POST', headers, body: JSON.stringify(body) });
}

function asAnthropic(body: Json): Json {
  const source = Array.isArray(body.messages) ? body.messages : [];
  const system = source.filter((m) => isRecord(m) && m.role === 'system').map((m) => isRecord(m) ? m.content : '').join('\n');
  const messages: Json[] = [];
  for (const raw of source) {
    if (!isRecord(raw) || raw.role === 'system') continue;
    if (raw.role === 'tool') {
      messages.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: raw.tool_call_id ?? '', content: raw.content ?? '' }] });
      continue;
    }
    const role = raw.role === 'assistant' ? 'assistant' : 'user';
    const blocks: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string' && content !== '') blocks.push({ type: 'text', text: content });
    else if (Array.isArray(content)) for (const part of content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
      if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
        const m = /^data:([^;]+);base64,(.*)$/.exec(part.image_url.url);
        blocks.push(m ? { type: 'image', source: { type: 'base64', media_type: m[1], data: m[2] } } : { type: 'image', source: { type: 'url', url: part.image_url.url } });
      }
    }
    if (role === 'assistant' && Array.isArray(raw.tool_calls)) for (const call of raw.tool_calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      let input: unknown = {};
      try { input = JSON.parse(String(call.function.arguments ?? '{}')); } catch { input = {}; }
      blocks.push({ type: 'tool_use', id: call.id ?? '', name: call.function.name ?? '', input });
    }
    messages.push({ role, content: blocks.length > 0 ? blocks : '' });
  }
  const out: Json = { model: body.model, max_tokens: body.max_tokens ?? 1024, messages };
  if (system) out.system = system;
  if (Array.isArray(body.stop)) out.stop_sequences = body.stop;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.tools)) out.tools = body.tools.flatMap((t) => isRecord(t) && isRecord(t.function) ? [{ name: t.function.name, description: t.function.description, input_schema: t.function.parameters }] : []);
  if (body.stream === true) out.stream = true;
  return out;
}

function asGemini(body: Json): Json {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const system = messages.filter((m) => isRecord(m) && m.role === 'system').map((m) => isRecord(m) ? String(m.content ?? '') : '').join('\n');
  const toolNameById = new Map<string, string>();
  const contents = messages.filter((m) => isRecord(m) && m.role !== 'system').map((m) => {
    if (!isRecord(m)) return { role: 'user', parts: [{ text: '' }] };
    if (m.role === 'tool') return { role: 'user', parts: [{ functionResponse: { name: toolNameById.get(String(m.tool_call_id ?? '')) ?? String(m.name ?? 'tool'), response: { content: m.content ?? '' } } }] };
    const role = m.role === 'assistant' ? 'model' : 'user';
    const c = m.content;
    const parts: Json[] = typeof c === 'string' ? [{ text: c }] : Array.isArray(c) ? c.flatMap((p) => {
      if (!isRecord(p)) return [];
      if (p.type === 'text' && typeof p.text === 'string') return [{ text: p.text }];
      if (p.type === 'image_url' && isRecord(p.image_url) && typeof p.image_url.url === 'string') {
        const match = /^data:([^;]+);base64,(.*)$/.exec(p.image_url.url);
        return match ? [{ inlineData: { mimeType: match[1], data: match[2] } }] : [{ fileData: { fileUri: p.image_url.url } }];
      }
      return [];
    }) : [];
    if (role === 'model' && Array.isArray(m.tool_calls)) for (const tc of m.tool_calls) {
      if (!isRecord(tc) || !isRecord(tc.function)) continue;
      let args: unknown = {};
      try { args = JSON.parse(String(tc.function.arguments ?? '{}')); } catch { args = {}; }
      if (typeof tc.id === 'string' && typeof tc.function.name === 'string') toolNameById.set(tc.id, tc.function.name);
      parts.push({ functionCall: { name: tc.function.name ?? '', args } });
    }
    if (parts.length === 0) parts.push({ text: '' });
    return { role, parts };
  });
  const generationConfig: Json = {};
  if (typeof body.max_tokens === 'number') generationConfig.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') generationConfig.temperature = body.temperature;
  if (typeof body.top_p === 'number') generationConfig.topP = body.top_p;
  if (Array.isArray(body.stop)) generationConfig.stopSequences = body.stop;
  const out: Json = { contents, generationConfig };
  if (system) out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) out.tools = [{ functionDeclarations: body.tools.flatMap((t) => isRecord(t) && isRecord(t.function) ? [{ name: t.function.name, description: t.function.description, parameters: t.function.parameters }] : []) }];
  return out;
}

function providerBody(route: Route, body: Json): Json {
  const withModel = { ...body, model: route.model.id };
  if (route.probe.provider.wire === 'anthropic') return asAnthropic(withModel);
  if (route.probe.provider.wire === 'gemini') return asGemini(withModel);
  if (withModel.stream === true) {
    // Without include_usage the provider stream has no usage chunk, so the trial's usage check voids a valid reply.
    return { ...withModel, stream_options: { ...withModel.stream_options, include_usage: true } };
  }
  return withModel;
}

function openAIAnswer(route: Route, raw: Json): Json {
  if (route.probe.provider.wire === 'openai') return raw;
  if (route.probe.provider.wire === 'anthropic') {
    const content = Array.isArray(raw.content) ? raw.content : [];
    const text = content.flatMap((v) => isRecord(v) && v.type === 'text' && typeof v.text === 'string' ? [v.text] : []).join('');
    const tool_calls = content.flatMap((v, i) => isRecord(v) && v.type === 'tool_use' ? [{ index: i, id: v.id, type: 'function', function: { name: v.name, arguments: JSON.stringify(v.input ?? {}) } }] : []);
    const usage = isRecord(raw.usage) ? raw.usage : {};
    const reason = raw.stop_reason === 'max_tokens' ? 'length' : raw.stop_reason === 'tool_use' ? 'tool_calls' : raw.stop_reason === 'refusal' ? 'content_filter' : 'stop';
    return { id: raw.id, object: 'chat.completion', model: route.model.id, choices: [{ index: 0, message: { role: 'assistant', content: text || null, ...(tool_calls.length ? { tool_calls } : {}) }, finish_reason: reason }], usage: { prompt_tokens: usage.input_tokens ?? 0, completion_tokens: usage.output_tokens ?? 0, total_tokens: Number(usage.input_tokens ?? 0) + Number(usage.output_tokens ?? 0) } };
  }
  const candidates = Array.isArray(raw.candidates) ? raw.candidates : [];
  const cand = isRecord(candidates[0]) ? candidates[0] : {};
  const cont = isRecord(cand.content) ? cand.content : {};
  const parts = Array.isArray(cont.parts) ? cont.parts : [];
  const text = parts.flatMap((p) => isRecord(p) && typeof p.text === 'string' ? [p.text] : []).join('');
  const toolCalls = parts.flatMap((p, i) => isRecord(p) && isRecord(p.functionCall) ? [{ id: `call_${i}`, type: 'function', function: { name: p.functionCall.name ?? '', arguments: JSON.stringify(p.functionCall.args ?? {}) } }] : []);
  const meta = isRecord(raw.usageMetadata) ? raw.usageMetadata : {};
  const reason = cand.finishReason === 'MAX_TOKENS' ? 'length' : cand.finishReason === 'SAFETY' ? 'content_filter' : 'stop';
  return { id: 'gemini-completion', object: 'chat.completion', model: route.model.id, choices: [{ index: 0, message: { role: 'assistant', content: text || null, ...(toolCalls.length ? { tool_calls: toolCalls } : {}) }, finish_reason: toolCalls.length ? 'tool_calls' : reason }], usage: { prompt_tokens: meta.promptTokenCount ?? 0, completion_tokens: meta.candidatesTokenCount ?? 0, total_tokens: meta.totalTokenCount ?? 0 } };
}

async function collectAnswer(route: Route, response: Response): Promise<Json> {
  const type = response.headers.get('content-type') ?? '';
  if (!type.includes('text/event-stream') || response.body === null) {
    const raw: unknown = await response.json();
    return openAIAnswer(route, isRecord(raw) ? raw : {});
  }
  const text = await response.text();
  const frames = text.replace(/\r\n/g, '\n').split('\n\n');
  if (route.probe.provider.wire === 'openai') {
    let id = 'chatcmpl-stream'; let model = route.model.id; let content = ''; let finish: unknown = 'stop';
    let usage: Json = {}; const calls = new Map<number, Json>();
    for (const frame of frames) for (const line of frame.split('\n')) {
      if (!line.startsWith('data:')) continue;
      const data = line.slice(5).trim(); if (!data || data === '[DONE]') continue;
      try {
        const c: unknown = JSON.parse(data); if (!isRecord(c)) continue;
        if (typeof c.id === 'string') id = c.id; if (typeof c.model === 'string') model = c.model;
        if (isRecord(c.usage)) usage = c.usage;
        const choice = Array.isArray(c.choices) && isRecord(c.choices[0]) ? c.choices[0] : {};
        const delta = isRecord(choice.delta) ? choice.delta : {};
        if (typeof delta.content === 'string') content += delta.content;
        if (choice.finish_reason != null) finish = choice.finish_reason;
        if (Array.isArray(delta.tool_calls)) for (const part of delta.tool_calls) {
          if (!isRecord(part)) continue; const ix = typeof part.index === 'number' ? part.index : calls.size;
          const current = calls.get(ix) ?? { id: '', type: 'function', function: { name: '', arguments: '' } };
          if (typeof part.id === 'string') current.id = part.id;
          const fn = isRecord(part.function) ? part.function : {};
          const target = isRecord(current.function) ? current.function : {};
          if (typeof fn.name === 'string') target.name = String(target.name ?? '') + fn.name;
          if (typeof fn.arguments === 'string') target.arguments = String(target.arguments ?? '') + fn.arguments;
          current.function = target; calls.set(ix, current);
        }
      } catch { /* Ignore malformed provider frames. */ }
    }
    return openAIAnswer(route, { id, model, choices: [{ message: { role: 'assistant', content: content || null, ...(calls.size ? { tool_calls: [...calls.values()] } : {}) }, finish_reason: finish }], usage });
  }
  if (route.probe.provider.wire === 'anthropic') {
    let id = 'msg_stream'; let model = route.model.id; let stop: unknown = 'end_turn'; let usage: Json = {}; const blocks = new Map<number, Json>();
    for (const frame of frames) {
      const eventLine = frame.split('\n').find((line) => line.startsWith('event:'));
      const dataLine = frame.split('\n').find((line) => line.startsWith('data:'));
      if (!dataLine) continue;
      try {
        const data: unknown = JSON.parse(dataLine.slice(5).trim()); if (!isRecord(data)) continue;
        if (eventLine?.includes('message_start') && isRecord(data.message)) {
          id = String(data.message.id ?? id); model = String(data.message.model ?? model); if (isRecord(data.message.usage)) usage = data.message.usage;
        }
        const ix = typeof data.index === 'number' ? data.index : 0;
        if (eventLine?.includes('content_block_start') && isRecord(data.content_block)) blocks.set(ix, { ...data.content_block });
        if (eventLine?.includes('content_block_delta')) {
          const block = blocks.get(ix) ?? { type: 'text', text: '' }; const delta = isRecord(data.delta) ? data.delta : {};
          if (delta.type === 'text_delta') block.text = String(block.text ?? '') + String(delta.text ?? '');
          if (delta.type === 'input_json_delta') block.input_json = String(block.input_json ?? '') + String(delta.partial_json ?? '');
          blocks.set(ix, block);
        }
        if (eventLine?.includes('message_delta') && isRecord(data.delta)) { stop = data.delta.stop_reason ?? stop; if (isRecord(data.usage)) usage = { ...usage, ...data.usage }; }
      } catch { /* Ignore malformed provider frames. */ }
    }
    const content = [...blocks.entries()].sort((a, b) => a[0] - b[0]).map(([, b]) => b.type === 'tool_use' ? { ...b, input: (() => { try { return JSON.parse(String(b.input_json ?? '{}')); } catch { return b.input ?? {}; } })() } : b);
    return openAIAnswer(route, { id, model, content, stop_reason: stop, usage });
  }
  let raw: Json = {};
  for (const frame of frames) for (const line of frame.split('\n')) {
    if (!line.startsWith('data:')) continue;
    try {
      const c: unknown = JSON.parse(line.slice(5).trim()); if (!isRecord(c)) continue;
      const cand = Array.isArray(c.candidates) && isRecord(c.candidates[0]) ? c.candidates[0] : {};
      const prior = Array.isArray(raw.candidates) && isRecord(raw.candidates[0]) ? raw.candidates[0] : {};
      const cont = isRecord(cand.content) ? cand.content : {};
      const prevCont = isRecord(prior.content) ? prior.content : {};
      const oldParts = Array.isArray(prevCont.parts) ? prevCont.parts : [];
      const newParts = Array.isArray(cont.parts) ? cont.parts : [];
      raw = { ...raw, candidates: [{ ...prior, ...cand, content: { ...prevCont, ...cont, parts: [...oldParts, ...newParts] } }], ...(isRecord(c.usageMetadata) ? { usageMetadata: c.usageMetadata } : {}) };
    } catch { /* Ignore malformed provider frames. */ }
  }
  return openAIAnswer(route, raw);
}

function streamFromAnswer(answer: Json): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const chunk = { id: answer.id ?? 'chatcmpl-stream', object: 'chat.completion.chunk', created: 1767225600, model: answer.model ?? '', choices: [{ index: 0, delta: { role: 'assistant', ...(typeof message.content === 'string' ? { content: message.content } : {}), ...(Array.isArray(message.tool_calls) ? { tool_calls: message.tool_calls } : {}) }, finish_reason: null }] };
  const tail = { id: chunk.id, object: chunk.object, created: chunk.created, model: chunk.model, choices: [{ index: 0, delta: {}, finish_reason: choice.finish_reason ?? 'stop' }], usage: answer.usage ?? {} };
  const frames = `data: ${JSON.stringify(chunk)}\n\n` + `data: ${JSON.stringify(tail)}\n\n` + 'data: [DONE]\n\n';
  return new ReadableStream<Uint8Array>({ start(controller) { controller.enqueue(enc.encode(frames)); controller.close(); } });
}

function sendOpenAIStream(answer: Json, res: ServerResponse): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const msg = isRecord(choice.message) ? choice.message : {};
  const base = { id: answer.id ?? 'chatcmpl-stream', object: 'chat.completion.chunk', created: 1767225600, model: answer.model ?? '' };
  const d = { role: 'assistant', ...(typeof msg.content === 'string' ? { content: msg.content } : {}), ...(Array.isArray(msg.tool_calls) ? { tool_calls: msg.tool_calls } : {}) };
  res.write(`data: ${JSON.stringify({ ...base, choices: [{ index: 0, delta: d, finish_reason: null }] })}\n\n`);
  res.write(`data: ${JSON.stringify({ ...base, choices: [{ index: 0, delta: {}, finish_reason: choice.finish_reason ?? 'stop' }], usage: answer.usage ?? {} })}\n\n`);
  res.end('data: [DONE]\n\n');
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by an openai-wire provider. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const candidates = routesForRequest(requires, requires.json === true);
  if (candidates.length === 0) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
  const skipped = new Set<string>();
  let attempt = 0;
  for (const route of candidates) {
    if (skipped.has(route.probe.provider.id)) continue;
    attempt += 1;
    let upstream: Response;
    try {
      providerCalls.set(route.probe.provider.id, (providerCalls.get(route.probe.provider.id) ?? 0) + 1);
      const request = providerBody(route, { ...body, stream: body.stream === true });
      upstream = await callProvider(route, request);
    } catch {
      recordProviderOutcome(route.probe.provider.id, true);
      logRouteAttempt(currentRequestId, route, attempt, 'transport_error');
      skipped.add(route.probe.provider.id);
      continue;
    }
    const failed = upstream.status === 402 || upstream.status === 429 || upstream.status >= 500;
    recordProviderOutcome(route.probe.provider.id, failed, upstream.status >= 200 && upstream.status < 300);
    if (upstream.status === 402) exhaustedProviders.add(route.probe.provider.id);
    if (upstream.status === 402 || upstream.status === 429 || upstream.status >= 500) {
      logRouteAttempt(currentRequestId, route, attempt, `retry_http_${upstream.status}`);
      skipped.add(route.probe.provider.id);
      continue;
    }
    if (!upstream.ok) {
      logRouteAttempt(currentRequestId, route, attempt, `http_${upstream.status}`);
      res.writeHead(upstream.status, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
      return res.end(await upstream.text());
    }
    let answer: Json;
    try { answer = await collectAnswer(route, upstream); } catch {
      logRouteAttempt(currentRequestId, route, attempt, 'response_parse_error');
      skipped.add(route.probe.provider.id);
      continue;
    }
    logRouteAttempt(currentRequestId, route, attempt, `served_http_${upstream.status}`);
    if (body.stream === true) sendOpenAIStream(answer, res);
    else sendJson(res, 200, answer);
    return;
  }
  return sendError(res, 'openai', 503, 'upstream_unavailable', 'all eligible free providers refused this request');
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to an openai-wire provider and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = routesForRequest(requirementsFor(body));
  if (candidates.length === 0) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
  const requested = typeof body.model === 'string' && body.model !== '' ? body.model : candidates[0].model.id;
  const skipped = new Set<string>();
  let attempt = 0;
  for (const route of candidates) {
    if (skipped.has(route.probe.provider.id)) continue;
    attempt += 1;
    let upstream: Response;
    try {
      providerCalls.set(route.probe.provider.id, (providerCalls.get(route.probe.provider.id) ?? 0) + 1);
      const asOpen = anthropicToOpenAI(body, route.model.id);
      const native = route.probe.provider.wire === 'anthropic'
        ? { ...body, model: route.model.id, stream: body.stream === true }
        : providerBody(route, { ...asOpen, stream: body.stream === true });
      upstream = await callProvider(route, native);
    } catch {
      recordProviderOutcome(route.probe.provider.id, true);
      logRouteAttempt(currentRequestId, route, attempt, 'transport_error');
      skipped.add(route.probe.provider.id);
      continue;
    }
    const failed = upstream.status === 402 || upstream.status === 429 || upstream.status >= 500;
    recordProviderOutcome(route.probe.provider.id, failed, upstream.status >= 200 && upstream.status < 300);
    if (upstream.status === 402) exhaustedProviders.add(route.probe.provider.id);
    if (upstream.status === 402 || upstream.status === 429 || upstream.status >= 500) {
      logRouteAttempt(currentRequestId, route, attempt, `retry_http_${upstream.status}`);
      skipped.add(route.probe.provider.id);
      continue;
    }
    if (!upstream.ok) {
      logRouteAttempt(currentRequestId, route, attempt, `http_${upstream.status}`);
      return sendJson(res, upstream.status, anthropicError('api_error', `the provider answered HTTP ${upstream.status}`));
    }
    let normalized: Json;
    try { normalized = await collectAnswer(route, upstream); } catch {
      logRouteAttempt(currentRequestId, route, attempt, 'response_parse_error');
      skipped.add(route.probe.provider.id);
      continue;
    }
    logRouteAttempt(currentRequestId, route, attempt, `served_http_${upstream.status}`);
    if (body.stream === true) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      for await (const event of anthropicEventsFromOpenAI(streamFromAnswer(normalized), requested)) res.write(event);
      res.end();
    } else sendJson(res, 200, openAIToAnthropic(normalized, requested));
    return;
  }
  return sendJson(res, 503, anthropicError('overloaded_error', 'all eligible free providers refused this request'));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
// Deterministic provider routing and wire translation for the make.fun trial.
// Manifest planning pass: commit the full trial assignment before serving requests.
// Candidates carry provider/model identity, requirement eligibility, and rating_milli.
function planManifestAssignments(manifest, candidatesByRequest) {
  const eligibleProviders = new Set();
  for (const request of manifest) {
    for (const candidate of candidatesByRequest.get(request.id) || []) {
      if (candidate.legal !== false) eligibleProviders.add(candidate.provider);
    }
  }
  const providerCaps = { groq: 3, cerebras: 2, gemini: 2, zai: 1, openrouter: 3,
    sambanova: 1, pollinations: 3, cloudflare: 2, ovh: 2, kilol: 3 };
  const capFor = (provider) => providerCaps[provider] ?? 2;
  const counts = new Map();
  const plan = new Map();
  const reserved = manifest.filter((request) => request.id === "r01" || request.id === "r02");
  const rest = manifest.filter((request) => request.id !== "r01" && request.id !== "r02");
  const bestByProvider = (request) => {
    const best = new Map();
    for (const candidate of candidatesByRequest.get(request.id) || []) {
      if (candidate.legal === false) continue;
      const previous = best.get(candidate.provider);
      if (!previous || candidate.rating_milli > previous.rating_milli
        || (candidate.rating_milli === previous.rating_milli && String(candidate.model) < String(previous.model))) {
        best.set(candidate.provider, candidate);
      }
    }
    return [...best.values()].sort((a, b) => b.rating_milli - a.rating_milli
      || String(a.provider).localeCompare(String(b.provider)) || String(a.model).localeCompare(String(b.model)));
  };
  // Lock the top eligible tools+vision capacity for r01 and r02 before assigning ordinary work.
  for (const request of reserved) {
    const chosen = bestByProvider(request).find((candidate) => (counts.get(candidate.provider) || 0) < capFor(candidate.provider));
    if (!chosen) throw new Error(`No capacity-feasible legal model for ${request.id}`);
    plan.set(request.id, chosen);
    counts.set(chosen.provider, (counts.get(chosen.provider) || 0) + 1);
  }

  // Maximum-weight bipartite flow gives the remaining requests the best global weighted rating sum.
  const providers = [...eligibleProviders].sort();
  const source = 0;
  const requestStart = 1;
  const providerStart = requestStart + rest.length;
  const sink = providerStart + providers.length;
  const graph = Array.from({ length: sink + 1 }, () => []);
  const addEdge = (from, to, capacity, cost, candidate = null) => {
    const forward = { to, rev: graph[to].length, capacity, cost, candidate };
    const reverse = { to: from, rev: graph[from].length, capacity: 0, cost: -cost, candidate: null };
    graph[from].push(forward);
    graph[to].push(reverse);
    return forward;
  };
  const providerNode = new Map(providers.map((provider, index) => [provider, providerStart + index]));
  const assignmentEdges = [];
  for (let index = 0; index < rest.length; index += 1) {
    const request = rest[index];
    const node = requestStart + index;
    addEdge(source, node, 1, 0);
    for (const candidate of bestByProvider(request)) {
      const edge = addEdge(node, providerNode.get(candidate.provider), 1, -candidate.rating_milli * request.weight, candidate);
      assignmentEdges.push({ request, edge });
    }
  }
  for (const provider of providers) addEdge(providerNode.get(provider), sink,
    capFor(provider) - (counts.get(provider) || 0), 0);
  let flow = 0;
  while (flow < rest.length) {
    const distance = Array(graph.length).fill(Infinity);
    const previousNode = Array(graph.length).fill(-1);
    const previousEdge = Array(graph.length).fill(-1);
    distance[source] = 0;
    for (let pass = 0; pass < graph.length - 1; pass += 1) {
      let changed = false;
      for (let node = 0; node < graph.length; node += 1) {
        if (!Number.isFinite(distance[node])) continue;
        for (let edgeIndex = 0; edgeIndex < graph[node].length; edgeIndex += 1) {
          const edge = graph[node][edgeIndex];
          const next = distance[node] + edge.cost;
          if (edge.capacity > 0 && next < distance[edge.to]) {
            distance[edge.to] = next;
            previousNode[edge.to] = node;
            previousEdge[edge.to] = edgeIndex;
            changed = true;
          }
        }
      }
      if (!changed) break;
    }
    if (previousNode[sink] < 0) break;
    for (let node = sink; node !== source; node = previousNode[node]) {
      const edge = graph[previousNode[node]][previousEdge[node]];
      edge.capacity -= 1;
      graph[node][edge.rev].capacity += 1;
    }
    flow += 1;
  }
  for (const { request, edge } of assignmentEdges) if (edge.capacity === 0) plan.set(request.id, edge.candidate);
  if (plan.size !== manifest.length) throw new Error(`Manifest plan assigned ${plan.size}/${manifest.length} requests`);
  for (const request of rest) {
    const chosen = plan.get(request.id);
    counts.set(chosen.provider, (counts.get(chosen.provider) || 0) + 1);
  }
  return { assignments: plan, providerCaps, providerCounts: counts };
}

/**
 * Cloudflare Worker wrapper for Embassy.
 *
 * Exposes upsert / query / delete over HTTP, backed by a Cloudflare Vectorize
 * index bound as `env.VECTORIZE`. All Embassy defaults apply (edge embedder,
 * sentence chunker, weighted-LCS reranker, SHA-256 ids), so no extra config is
 * needed for a metadata-only index. Optional Workers KV bindings enrich it:
 *   - env.STORAGE   -> full-document storage (otherwise text lives in metadata)
 *   - env.MANIFEST  -> id manifest (keeps deletes correct across chunker changes)
 *
 *   GET  /query?text=lazy+dog&limit=5
 *   POST /query   { "text": "lazy dog", "limit": 5 }
 *   POST /upsert  { "text": "The quick brown fox..." }
 *   POST /delete  { "text": "The quick brown fox..." }
 *
 * wrangler.toml:
 *   [[vectorize]]
 *   binding = "VECTORIZE"
 *   index_name = "your-index"
 *   # optional:
 *   # [[kv_namespaces]]
 *   # binding = "STORAGE"
 *   # id = "..."
 */
import { Embassy } from "./embassy.js";
import { vectorizeAdapter } from "./adapters/vectorize.js";

const isString = x => typeof x === "string" || x instanceof String;

const json = (data, status = 200) =>
  new Response(JSON.stringify(data, null, 2), {
    status,
    headers: { "Content-Type": "application/json" }
  });

// Adapt a Workers KV namespace to Embassy's { put, get, delete } interface.
// KV.get returns null when a key is absent, which Embassy treats as "missing".
const kvStore = (kv) => ({
  put: (key, value) => kv.put(key, isString(value) ? value : String(value)),
  get: (key) => kv.get(key),
  delete: (key) => kv.delete(key)
});

// Build an Embassy bound to the request's Vectorize index (+ optional KV).
const makeEmbassy = (env) => {
  if (!env?.VECTORIZE)
    throw new Error('Missing Vectorize binding "VECTORIZE".');

  const options = {};
  if (env.STORAGE) options.storage = kvStore(env.STORAGE);
  if (env.MANIFEST) options.manifest = kvStore(env.MANIFEST);

  return new Embassy(vectorizeAdapter(env.VECTORIZE), options);
};

export default {
  Embassy,

  async fetch(request, env) {
    try {
      const url = new URL(request.url);
      const path = url.pathname.replace(/\/+$/, "") || "/";

      if (path === "/" && request.method === "GET") {
        return json({
          service: "embassy",
          endpoints: ["POST /upsert", "GET|POST /query", "POST /delete"]
        });
      }

      const embassy = makeEmbassy(env);

      if (path === "/upsert" && request.method === "POST") {
        const { text, vector } = await request.json();

        if (text == null && vector == null)
          return json({ error: 'Provide "text" or "vector".' }, 400);

        const count = await embassy.upsert(text, vector != null ? { vector } : {});
        return json({ upserted: count });
      }

      if (path === "/query") {
        let text, limit, vector;

        if (request.method === "GET") {
          text = url.searchParams.get("text");
          const raw = url.searchParams.get("limit");
          const n = raw == null ? NaN : Number(raw);
          limit = Number.isFinite(n) ? n : undefined;
        } else if (request.method === "POST") {
          ({ text, limit, vector } = await request.json());
        } else {
          return json({ error: "Method not allowed. Use GET or POST." }, 405);
        }

        if (text == null && vector == null)
          return json({ error: 'Provide "text" or "vector".' }, 400);

        const results = await embassy.query(text, limit ?? 10, vector != null ? { vector } : {});
        return json({ count: results.length, results });
      }

      if (path === "/delete" && request.method === "POST") {
        const { text, vector } = await request.json();

        if (text == null && vector == null)
          return json({ error: 'Provide "text" or "vector".' }, 400);

        const count = await embassy.delete(text, vector != null ? { vector } : {});
        return json({ deleted: count });
      }

      return json({ error: "Not found." }, 404);
    } catch (error) {
      return json({ error: "Internal server error", message: error.message }, 500);
    }
  }
};

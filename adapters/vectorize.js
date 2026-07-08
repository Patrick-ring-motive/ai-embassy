/**
 * Cloudflare Vectorize adapter.
 *
 * Maps the Pinecone-style interface Embassy emits onto the Vectorize Workers
 * binding (env.YOUR_INDEX). Usage:
 *
 *   import { vectorizeAdapter } from "./adapters/vectorize.js";
 *   const embassy = new Embassy(vectorizeAdapter(env.PROD_SEARCH), { ... });
 *
 * Embassy interface -> Vectorize binding:
 *   upsert(records)                          -> index.upsert(records)
 *   query({ vector, topK, includeMetadata }) -> index.query(vector, { topK, returnMetadata })
 *   deleteMany(ids)                          -> index.deleteByIds(ids)
 *
 * Notes:
 * - Records share the same shape ({ id, values, metadata }), so upsert is a
 *   passthrough. Vectorize upsert/delete are asynchronous (eventual
 *   consistency): the returned mutationId settles in a few seconds, so a query
 *   issued immediately after a write may not see it yet.
 * - Vectorize caps topK at 50 when returnMetadata is "all" (or returnValues is
 *   true) and at 100 otherwise, so topK is clamped here to avoid an error. If
 *   your metadata is small and indexed, switch returnMetadata to "indexed" to
 *   allow topK up to 100.
 */
export const vectorizeAdapter = (index) => ({
  async upsert(records) {
    return index.upsert(records);
  },

  async query({
    vector,
    topK = 10,
    includeMetadata = false,
    includeValues = false,
    filter
  }) {
    const returnMetadata = includeMetadata ? "all" : "none";
    const cap = includeMetadata || includeValues ? 50 : 100;

    // Returns { count, matches: [{ id, score, values?, metadata? }] };
    // Embassy reads .matches.
    return index.query(vector, {
      topK: Math.min(topK, cap),
      returnMetadata,
      returnValues: includeValues,
      ...(filter ? {
        filter
      } : {})
    });
  },

  async deleteMany(ids) {
    return index.deleteByIds(ids);
  }
});

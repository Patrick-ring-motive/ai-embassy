import edgeEmbeddingWorker from '../edge-embedding/embed.js';
/* https://github.com/Patrick-ring-motive/edge-embedding/blob/main/embed.js */
import { rank } from '../weighted-lcs-reranker/reranker.js';
/* https://github.com/Patrick-ring-motive/weighted-lcs-reranker/blob/main/reranker.js */
import defaultChunker from '../sentence-chunker/chunker.js';
import { hash } from '../vector-hash/hash.js';
const isArray = x => Array.isArray(x) || x instanceof Array;
const isString = x => typeof x === 'string' || x instanceof String;
const edgeEmbed = edgeEmbeddingWorker.edgeEmbed;
const stringify = x => {
  try {
    if (isString(x)) {
      return String(x);
    }
    return String(JSON.stringify(x));
  } catch {
    return String(x);
  }
};

// Truncate a string so its UTF-8 length is at most `maxBytes`, cutting on a
// codepoint boundary (never splitting a character). Appends "..." when it
// actually truncates, keeping the total within `maxBytes`. Byte-based because
// vector DB metadata limits (e.g. Vectorize's ~10 KiB) are measured in bytes,
// not UTF-16 code units.
const truncateToBytes = (text, maxBytes) => {
  const encoder = new TextEncoder();
  const str = String(text);

  if (maxBytes <= 0) return "";
  if (encoder.encode(str).length <= maxBytes) return str;

  const suffix = "...";
  const budget = Math.max(0, maxBytes - encoder.encode(suffix).length);
  const { read } = encoder.encodeInto(str, new Uint8Array(budget));

  return str.slice(0, read) + suffix;
};

// Caches the detected embedding dimension per embedder object, so every
// Embassy sharing an embedder (including the shared default) probes at most
// once. A failed probe is evicted so a later call can retry.
const vectorSizeByEmbedder = new WeakMap();

// Cloudflare Vectorize's hard cap on topK. It also stops returning metadata
// for results beyond 50, which is why candidateLimit defaults to 50.
const MAX_CANDIDATES = 100;

export class Embassy {
  static defaults = {
    /**
     * Embedding provider.
     *
     * Required interface:
     *   await embed(texts:string[]) => number[][]
     *
     * Optional shorthand:
     *   await embed(text:string) => number[]
     */
    embedder: {
      async embed(texts) {
        if (isArray(texts)) {
          return texts.map(edgeEmbed);
        }
        return edgeEmbed(stringify(texts));
      }
    },

    /**
     * Optional embedding dimensionality.
     *
     * When set, vectors supplied via the { vector } option are normalized to
     * this length and no probe embedding is performed. Otherwise the size is
     * taken from embedder.vectorSize, or detected once per embedder by
     * embedding an empty string on first use.
     */
    vectorSize: null,

    /**
     * Optional document storage.
     *
     * Interface:
     *   put(id, text)
     *   get(id)
     *   delete(id)
     */
    storage: null,

    /**
     * Optional id manifest.
     *
     * Records the exact vector ids written for each document so delete() can
     * remove them without re-running the chunker; this keeps deletes correct
     * even if the chunker changes between upsert and delete. Same interface as
     * storage (keys are derived from the document text):
     *   put(key, ids)   // ids is a serialized string
     *   get(key)        // -> the stored string, or null/undefined if absent
     *   delete(key)
     * When absent — or when a specific entry is missing — delete() falls back
     * to re-deriving the ids from the chunker.
     */
    manifest: null,

    /**
     * Optional chunker.
     *
     * Interface:
     *   chunk(text) => [{ text, metadata? }]
     */
    chunker: defaultChunker,

    /**
     * Optional reranker.
     *
     * Interface:
     *   rank(query, results) => results
     */
    reranker: {
      rank
    },

    /**
     * Number of candidates to fetch before reranking, then trimmed to the
     * requested limit. Only applied when a reranker actually runs. Defaults
     * to 50 (Vectorize returns no metadata beyond 50) and is hard-capped at
     * MAX_CANDIDATES (100).
     */
    candidateLimit: 50,

    /**
     * Hash function used as document id. Defaults to SHA-256 hex (see
     * ../hash/hash.js); called as a plain function so it can be swapped.
     */
    hash,

    /**
     * Max UTF-8 byte length for text embedded in vector metadata when no
     * storage is configured. Kept under Vectorize's ~10 KiB metadata limit,
     * leaving headroom for other fields.
     */
    maxMetadataBytes: 8192,

    /**
     * Used when storage isn't configured. Truncates on a UTF-8 byte budget
     * so multi-byte text can't overflow the vector DB's metadata limit.
     */
    metadata(text, maxBytes = 8192) {
      return {
        text: truncateToBytes(text, maxBytes)
      };
    }
  };

  constructor(vectordb, options = {}) {
    if (!vectordb)
      throw new Error("Vector database is required.");

    for (const method of ["upsert", "query", "deleteMany"]) {
      if (typeof vectordb[method] !== "function") {
        throw new Error(`Vector database must implement ${method}().`);
      }
    }

    this.vectordb = vectordb;

    this.options = {
      ...Embassy.defaults,
      ...options
    };
  }

  async embed(texts) {
    const {
      embedder
    } = this.options;

    let vectors = await embedder.embed(texts);

    // Allow the embed(string) shorthand by wrapping a single flat vector as
    // one row, but leave an empty result ([]) alone so embed([]) stays [].
    if (vectors.length && !isArray(vectors[0]))
      vectors = [vectors];

    return vectors;
  }

  /**
   * Resolve the embedding dimension used to normalize caller-supplied
   * vectors. Precedence: options.vectorSize, then embedder.vectorSize, then
   * a one-time probe (embed of an empty string) memoized per embedder.
   */
  #getVectorSize() {
    const { embedder, vectorSize } = this.options;

    if (vectorSize != null)
      return Promise.resolve(vectorSize);

    if (embedder?.vectorSize != null)
      return Promise.resolve(embedder.vectorSize);

    let detected = vectorSizeByEmbedder.get(embedder);

    if (!detected) {
      detected = this.embed([""]).then(([vector]) => vector.length);

      vectorSizeByEmbedder.set(embedder, detected);

      // Evict a failed probe so a later call can retry; also prevents an
      // unhandled rejection when nothing awaits it before it settles.
      detected.catch(() => vectorSizeByEmbedder.delete(embedder));
    }

    return detected;
  }

  /**
   * Pad (with zeros) or truncate a caller-supplied vector so its length
   * matches the baseline embedding dimension.
   */
  async #normalizeVectorLength(vector) {
    const source = isArray(vector) ? vector : Array.from(vector ?? []);
    const length = await this.#getVectorSize();

    if (source.length === length)
      return source;

    if (source.length > length)
      return source.slice(0, length);

    return source.concat(new Array(length - source.length).fill(0));
  }

  #normalizeChunks(chunks) {
    if (!isArray(chunks)) {
      throw new Error("Chunker must return an array of chunks.");
    }

    return chunks.map((chunk, i) => {
      if (!chunk || typeof chunk !== "object") {
        throw new Error(`Invalid chunk at index ${i}; expected an object.`);
      }

      const chunkText = isString(chunk.text) ? String(chunk.text) : stringify(chunk.text);

      if (!chunkText) {
        throw new Error(`Invalid chunk at index ${i}; missing text.`);
      }

      const chunkMetadata =
        chunk.metadata && typeof chunk.metadata === "object" ? chunk.metadata : {};

      return {
        ...chunk,
        text: chunkText,
        metadata: chunkMetadata
      };
    });
  }

  // Chunk `text` and derive the vector-db id for each chunk. Single source of
  // truth for the id scheme so upsert() and delete() can't drift:
  //   single chunk    -> hash(chunk.text)          (content-addressed; dedups)
  //   multiple chunks -> `${hash(text)}:${index}`   (positional)
  // Correct deletion therefore requires a deterministic chunker: it must
  // produce the same chunks for the same input at delete time as at upsert
  // time, otherwise the recomputed ids won't match and vectors are orphaned.
  async #chunkAndIds(text) {
    const { chunker, hash } = this.options;

    const rawChunks = await chunker.chunk(text);
    const chunks = this.#normalizeChunks(rawChunks);

    if (chunks.length === 1) {
      return { chunks, ids: [await hash(chunks[0].text)] };
    }

    const textHash = await hash(text);
    const ids = chunks.map((_, i) => `${textHash}:${i}`);

    return { chunks, ids };
  }

  // Manifest key for a document, derivable from the text alone (no chunker) so
  // delete() can look up the recorded ids even if chunking has since changed.
  async #manifestKey(text) {
    const { hash } = this.options;
    return `manifest:${await hash(text)}`;
  }

  // Reads the ids previously recorded under a manifest key. Returns an array of
  // ids, or undefined when the entry is absent or not a usable array.
  async #readManifestIds(manifestKey) {
    const recorded = await this.options.manifest.get(manifestKey);

    if (recorded == null)
      return undefined;

    const parsed = JSON.parse(recorded);
    return isArray(parsed) ? parsed : undefined;
  }

  async upsert(text, { vector } = {}) {
    const {
      storage,
      hash,
      metadata,
      maxMetadataBytes,
      manifest
    } = this.options;

    // Upsert a caller-supplied vector directly, bypassing chunking/embedding.
    if (vector != null) {
      const values = await this.#normalizeVectorLength(vector);
      const id = await hash(stringify(values));

      await this.vectordb.upsert([{
        id,
        values,
        metadata: storage ? { id } : {}
      }]);

      return 1;
    }

    const { chunks, ids } = await this.#chunkAndIds(text);

    // When a manifest is configured, look up what a previous upsert of this
    // same text recorded, so we can clean up any ids the new chunking no longer
    // covers (read before embedding so a manifest failure fails fast).
    let manifestKey;
    let previousIds;

    if (manifest) {
      manifestKey = await this.#manifestKey(text);
      previousIds = await this.#readManifestIds(manifestKey);
    }

    const vectors = await this.embed(
      chunks.map(c => c.text)
    );

    const entries = [];

    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      const id = ids[i];

      let meta = chunk.metadata;

      if (storage) {
        await storage.put(id, chunk.text);

        meta = {
          ...meta,
          id
        };
      } else {
        meta = {
          ...meta,
          ...metadata(chunk.text, maxMetadataBytes)
        };
      }

      entries.push({
        id,
        values: vectors[i],
        metadata: meta
      });
    }

    await this.vectordb.upsert(entries);

    if (manifest) {
      // Remove vectors from a previous upsert that the new chunking no longer
      // produces (e.g. after a chunker change), so they don't linger as
      // orphans. Ids still present are left alone — the upsert above already
      // refreshed them.
      if (previousIds) {
        const current = new Set(ids);
        const stale = previousIds.filter(id => !current.has(id));

        if (stale.length) {
          await this.vectordb.deleteMany(stale);

          if (storage)
            await Promise.all(stale.map(id => storage.delete(id)));
        }
      }

      // Record the exact ids so delete() can find them without re-chunking.
      await manifest.put(manifestKey, JSON.stringify(ids));
    }

    return entries.length;
  }

  async query(text, limit = 10, { vector } = {}) {
    const args = [...arguments];
    const {
      storage,
      reranker,
      candidateLimit
    } = this.options;

    // Query by a caller-supplied vector directly, or embed the text.
    let queryVector;

    if (vector != null) {
      queryVector = await this.#normalizeVectorLength(vector);
    } else {
      [queryVector] = await this.embed([text]);
    }

    // Reranking needs a text query; skip it for vector-only queries.
    const willRerank = Boolean(reranker?.rank && text != null);

    // Over-fetch a candidate pool only when a reranker will reorder it, so it
    // can surface matches the ANN search ranked just below the requested limit.
    const poolSize = willRerank ?
      Math.min(Math.max(limit, candidateLimit), MAX_CANDIDATES) :
      Math.min(limit, MAX_CANDIDATES);

    const { matches = [] } = await this.vectordb.query({
      vector: queryVector,
      topK: poolSize,
      includeMetadata: true
    });

    const settled = await Promise.allSettled(
      matches.map(async result => {
        const resultMetadata = result?.metadata || {};
        let document;

        if (storage) {
          if (resultMetadata.id == null) {
            // No id to look up. Rather than abort the whole query, warn and
            // fall back to the stringified metadata we already have. Not
            // recommended in general, but handy for small/side-project
            // indexes that may contain foreign or id-less vectors.
            console.warn("Embassy.query: vector result missing metadata.id; falling back to stringified metadata.", result,...args);
            document = stringify(resultMetadata);
          } else {
            document = await storage.get(resultMetadata.id);
          }
        } else {
          document = resultMetadata.text;
        }

        // Missing/absent documents become an empty string rather than the
        // literal "null"/"undefined" that stringify would otherwise produce.
        if (document == null) {
          document = "";
        } else if (!isString(document)) {
          document = stringify(document);
        }

        return {
          ...result,
          text: document
        };
      })
    );

    // Keep the query resilient: a failed document lookup (e.g. a storage.get
    // rejection) drops only that result instead of failing the whole query.
    for (const outcome of settled) {
      if (outcome.status === "rejected") {
        console.warn("Embassy.query: dropping a result whose document could not be resolved.", outcome.reason);
      }
    }

    let results = settled
      .filter(outcome => outcome.status === "fulfilled")
      .map(outcome => outcome.value);

    // Reranking needs a text query; skip it for vector-only queries.
    if (willRerank)
      results = await reranker.rank(text, results);

    // Trim the (possibly over-fetched) candidate pool to the requested limit.
    return results.slice(0, limit);
  }

  async delete(text, { vector } = {}) {
    const {
      storage,
      hash,
      manifest
    } = this.options;

    // Delete a caller-supplied vector directly; id must match upsert(vector).
    if (vector != null) {
      const values = await this.#normalizeVectorLength(vector);
      const id = await hash(stringify(values));

      await this.vectordb.deleteMany([id]);

      if (storage)
        await storage.delete(id);

      return 1;
    }

    // Prefer the recorded id manifest so deletion stays correct even if the
    // chunker has changed since upsert; fall back to re-deriving the ids.
    let ids;
    let manifestKey;

    if (manifest) {
      manifestKey = await this.#manifestKey(text);
      ids = await this.#readManifestIds(manifestKey);
    }

    if (!ids)
      ({ ids } = await this.#chunkAndIds(text));

    await this.vectordb.deleteMany(ids);

    if (storage) {
      await Promise.all(
        ids.map(id => storage.delete(id))
      );
    }

    if (manifest)
      await manifest.delete(manifestKey);

    return ids.length;
  }
}

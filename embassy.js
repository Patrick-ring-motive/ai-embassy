import edgeEmbeddingWorker from '../edge-embedding/embed.js';
/* https://github.com/Patrick-ring-motive/edge-embedding/blob/main/embed.js */
import { rank } from '../weighted-lcs-reranker/reranker.js';
/* https://github.com/Patrick-ring-motive/weighted-lcs-reranker/blob/main/reranker.js */
import defaultChunker from '../sentence-chunker/chunker.js';
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
     * Hash function used as document id.
     */
    async hash(text) {
      const bytes = new TextEncoder().encode(text);
      const digest = await crypto.subtle.digest("SHA-256", bytes);

      return [...new Uint8Array(digest)]
        .map(b => b.toString(16).padStart(2, "0"))
        .join("");
    },

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

    // allow embed(string) shorthand
    if (!isArray(vectors[0]))
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

  async upsert(text, { vector } = {}) {
    const {
      chunker,
      storage,
      hash,
      metadata,
      maxMetadataBytes
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

    const rawChunks = await chunker.chunk(text);
    const chunks = this.#normalizeChunks(rawChunks);

    const vectors = await this.embed(
      chunks.map(c => c.text)
    );

    // ID scheme (must match delete()):
    //   single chunk    -> hash(chunk.text)
    //   multiple chunks -> `${hash(text)}:${index}`
    // Relies on the chunker producing the same number of chunks for the
    // same input in both upsert() and delete().
    const textHash = await hash(text);

    const entries = [];

    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];

      const id =
        chunks.length === 1 ?
        await hash(chunk.text) :
        `${textHash}:${i}`;

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

    let results = await Promise.all(
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

    // Reranking needs a text query; skip it for vector-only queries.
    if (willRerank)
      results = await reranker.rank(text, results);

    // Trim the (possibly over-fetched) candidate pool to the requested limit.
    return results.slice(0, limit);
  }

  async delete(text, { vector } = {}) {
    const {
      chunker,
      storage,
      hash
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

    const rawChunks = await chunker.chunk(text);
    const chunks = this.#normalizeChunks(rawChunks);
    const textHash = await hash(text);

    // ID scheme must match upsert(): single -> hash(chunk.text),
    // multiple -> `${hash(text)}:${index}`.
    const ids =
      chunks.length === 1 ? [await hash(chunks[0].text)] :
      chunks.map((_, i) => `${textHash}:${i}`);

    await this.vectordb.deleteMany(ids);

    if (storage) {
      await Promise.all(
        ids.map(id => storage.delete(id))
      );
    }

    return ids.length;
  }
}

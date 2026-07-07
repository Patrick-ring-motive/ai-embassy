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
     * Used when storage isn't configured.
     */
    metadata(text) {
      return {
        text: text.length > 8192 ?
          text.slice(0, 8189) + "..." : text
      };
    }
  };

  constructor(vectordb, options = {}) {
    if (!vectordb)
      throw new Error("Vector database is required.");

    this.vectordb = vectordb;

    this.options = {
      ...Embassy.defaults,
      ...options
    };
  }

  async #embed(texts) {
    const {
      embedder
    } = this.options;

    let vectors = await embedder.embed(texts);

    // allow embed(string) shorthand
    if (!isArray(vectors[0]))
      vectors = [vectors];

    return vectors;
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

  async upsert(text) {
    const {
      chunker,
      storage,
      hash,
      metadata
    } = this.options;

    const rawChunks = await chunker.chunk(text);
    const chunks = this.#normalizeChunks(rawChunks);

    const vectors = await this.#embed(
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
          ...metadata(chunk.text)
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

  async query(text, limit = 10) {
    const {
      storage,
      reranker
    } = this.options;

    const [vector] = await this.#embed([text]);

    let results = await this.vectordb.query(vector, limit);

    results = await Promise.all(
      results.map(async result => {
        const resultMetadata = result?.metadata || {};
        let document;

        if (storage) {
          if (resultMetadata.id == null) {
            throw new Error("Vector result is missing metadata.id required by storage.");
          }
          document = await storage.get(resultMetadata.id);
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

    if (reranker?.rank)
      results = await reranker.rank(text, results);

    return results;
  }

  async delete(text) {
    const {
      chunker,
      storage,
      hash
    } = this.options;

    const rawChunks = await chunker.chunk(text);
    const chunks = this.#normalizeChunks(rawChunks);
    const textHash = await hash(text);

    // ID scheme must match upsert(): single -> hash(chunk.text),
    // multiple -> `${hash(text)}:${index}`.
    const ids =
      chunks.length === 1 ? [await hash(chunks[0].text)] :
      chunks.map((_, i) => `${textHash}:${i}`);

    await this.vectordb.delete(ids);

    if (storage) {
      await Promise.all(
        ids.map(id => storage.delete(id))
      );
    }

    return ids.length;
  }
}

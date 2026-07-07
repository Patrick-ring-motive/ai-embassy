import edgeEmbed from 'embed.js';
/* https://github.com/Patrick-ring-motive/edge-embedding/blob/main/embed.js */
import rank from 'reranker.js';
/* https://github.com/Patrick-ring-motive/weighted-lcs-reranker/blob/main/reranker.js */
const isArray = x => Array.isArray(x) || x instanceof Array;
const isString = x => typeof x === 'string' || x instanceof String;
const stringify = x => {
  try {
    if (isString(x)) {
      return String(x);
    }
    return JSON.stringify(x);
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
    chunker: {
      async chunk(text) {
        return [{
          text
        }];
      }
    },

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
          text.slice(0, 8189) + "..." :
          text
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

  async upsert(text) {
    const {
      chunker,
      storage,
      hash,
      metadata
    } = this.options;

    const chunks = await chunker.chunk(text);

    const vectors = await this.#embed(
      chunks.map(c => c.text)
    );

    const entries = [];

    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];

      const id =
        chunks.length === 1 ?
        await hash(chunk.text) :
        `${await hash(text)}:${i}`;

      let meta = chunk.metadata || {};

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
        let document;

        if (storage)
          document = await storage.get(result.metadata.id);
        else
          document = result.metadata.text;

        return {
          ...result,
          text: document
        };
      })
    );

    if (reranker)
      results = await reranker.rank(text, results);

    return results;
  }

  async delete(text) {
    const {
      chunker,
      storage,
      hash
    } = this.options;

    const chunks = await chunker.chunk(text);

    const ids =
      chunks.length === 1 ?
      [await hash(text)] :
      await Promise.all(
        chunks.map((_, i) =>
          hash(text).then(h => `${h}:${i}`)
        )
      );

    await this.vectordb.delete(ids);

    if (storage) {
      await Promise.all(
        ids.map(id => storage.delete(id))
      );
    }

    return ids.length;
  }
}

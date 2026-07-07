/**
 * Vector DB adapters.
 *
 * Embassy speaks one "common" (Pinecone-style) interface and each adapter
 * translates it to a specific backend. An adapter implements three methods:
 *
 *   upsert(records)
 *     records: [{ id: string, values: number[], metadata?: object }]
 *
 *   query({ vector, topK, includeMetadata, includeValues?, filter? })
 *     -> { matches: [{ id: string, score: number, metadata?: object, values?: number[] }] }
 *
 *   deleteMany(ids: string[])
 *
 * Score conventions differ by backend: Pinecone / Vectorize / Milvus return a
 * similarity (higher = closer), Qdrant depends on the configured metric, and
 * Chroma / Weaviate / pgvector return a distance (lower = closer). Embassy's
 * reranker recomputes its own score from text, so this only matters if you read
 * result.score directly on a vector-only query.
 */

// Cloudflare Vectorize lives in its own file so it can be edited/imported
// independently.
export { vectorizeAdapter } from "./vectorize.js";

/**
 * Pinecone (@pinecone-database/pinecone). Pass an index or namespace handle,
 * e.g. pinecone.index("docs") or pinecone.index("docs").namespace("ns").
 *
 * Embassy already speaks Pinecone's shape, so this is effectively a passthrough.
 */
export const pineconeAdapter = (index) => ({
  async upsert(records) {
    return index.upsert(records);
  },

  async query({ vector, topK = 10, includeMetadata = false, includeValues = false, filter }) {
    return index.query({
      vector,
      topK,
      includeMetadata,
      includeValues,
      ...(filter ? { filter } : {})
    });
  },

  async deleteMany(ids) {
    return index.deleteMany(ids);
  }
});

/**
 * Qdrant (@qdrant/js-client-rest). Pass the client and a collection name.
 *
 * Caveat: Qdrant point IDs must be unsigned integers or UUIDs. Embassy's
 * content-hash / `${hash}:${i}` ids will not validate as-is; map them to UUIDs
 * (or keep the hash in the payload) before using this in production.
 */
export const qdrantAdapter = (client, collection) => ({
  async upsert(records) {
    return client.upsert(collection, {
      wait: true,
      points: records.map(r => ({
        id: r.id,
        vector: Array.from(r.values),
        payload: r.metadata ?? {}
      }))
    });
  },

  async query({ vector, topK = 10, includeMetadata = false, filter }) {
    // client.search returns ScoredPoint[]: { id, score, payload, vector }.
    const points = await client.search(collection, {
      vector: Array.from(vector),
      limit: topK,
      with_payload: includeMetadata,
      ...(filter ? { filter } : {})
    });

    return {
      matches: points.map(p => ({ id: p.id, score: p.score, metadata: p.payload ?? {} }))
    };
  },

  async deleteMany(ids) {
    return client.delete(collection, { points: ids });
  }
});

/**
 * Milvus (@zilliz/milvus2-sdk-node). Pass the client, a collection name, and
 * (optionally) the field names in your schema.
 *
 * Caveat: Milvus requires a predefined schema. Store arbitrary metadata in a
 * single JSON field (metadataField) or map it to typed columns.
 */
export const milvusAdapter = (client, collectionName, {
  idField = "id",
  vectorField = "vector",
  metadataField = "metadata",
  outputFields = [idField, metadataField]
} = {}) => ({
  async upsert(records) {
    return client.upsert({
      collection_name: collectionName,
      data: records.map(r => ({
        [idField]: r.id,
        [vectorField]: Array.from(r.values),
        [metadataField]: r.metadata ?? {}
      }))
    });
  },

  async query({ vector, topK = 10, includeMetadata = false, filter }) {
    const res = await client.search({
      collection_name: collectionName,
      data: [Array.from(vector)],
      limit: topK,
      output_fields: includeMetadata ? outputFields : [idField],
      ...(filter ? { filter } : {})
    });

    // res.results: [{ [idField], score, [metadataField]?, ... }]
    return {
      matches: (res.results ?? []).map(row => ({
        id: row[idField],
        score: row.score,
        metadata: row[metadataField] ?? {}
      }))
    };
  },

  async deleteMany(ids) {
    return client.delete({ collection_name: collectionName, ids });
  }
});

/**
 * Chroma (chromadb). Pass a collection handle from client.getCollection(...).
 *
 * Chroma is columnar and nests one row per query embedding; scores are
 * distances (lower = closer).
 */
export const chromaAdapter = (collection) => ({
  async upsert(records) {
    return collection.upsert({
      ids: records.map(r => r.id),
      embeddings: records.map(r => Array.from(r.values)),
      metadatas: records.map(r => r.metadata ?? {})
    });
  },

  async query({ vector, topK = 10, includeMetadata = false, filter }) {
    const res = await collection.query({
      queryEmbeddings: [Array.from(vector)],
      nResults: topK,
      include: includeMetadata ? ["metadatas", "distances"] : ["distances"],
      ...(filter ? { where: filter } : {})
    });

    const ids = res.ids?.[0] ?? [];
    const distances = res.distances?.[0] ?? [];
    const metadatas = res.metadatas?.[0] ?? [];

    return {
      matches: ids.map((id, i) => ({
        id,
        score: distances[i],
        metadata: metadatas[i] ?? {}
      }))
    };
  },

  async deleteMany(ids) {
    return collection.delete({ ids });
  }
});

/**
 * Weaviate (weaviate-client v3). Pass a collection handle from
 * client.collections.get(...).
 *
 * Caveats: object IDs must be UUIDs (same as Qdrant); insertMany does not
 * overwrite existing IDs (use replace() for true upsert); nearVector metadata
 * exposes a distance (lower = closer), not a similarity score.
 */
export const weaviateAdapter = (collection) => ({
  async upsert(records) {
    return collection.data.insertMany(
      records.map(r => ({
        id: r.id,
        vector: Array.from(r.values),
        properties: r.metadata ?? {}
      }))
    );
  },

  async query({ vector, topK = 10, includeMetadata = false, filter }) {
    const res = await collection.query.nearVector(Array.from(vector), {
      limit: topK,
      returnMetadata: ["distance"],
      ...(includeMetadata ? {} : { returnProperties: [] }),
      ...(filter ? { filters: filter } : {})
    });

    // res.objects: [{ uuid, properties, metadata: { distance } }]
    return {
      matches: (res.objects ?? []).map(o => ({
        id: o.uuid,
        score: o.metadata?.distance,
        metadata: o.properties ?? {}
      }))
    };
  },

  async deleteMany(ids) {
    return collection.data.deleteMany(
      collection.filter.byId().containsAny(ids)
    );
  }
});

/**
 * pgvector (node-postgres). Pass a Pool/Client and, optionally, your table and
 * column names.
 *
 * SECURITY: table/column names are interpolated into SQL because identifiers
 * cannot be parameterized. Keep them as trusted, developer-controlled config —
 * never derive them from user input. All row *values* are passed as bound
 * parameters.
 */
export const pgvectorAdapter = (pool, {
  table = "embeddings",
  idColumn = "id",
  vectorColumn = "embedding",
  metadataColumn = "metadata",
  metric = "cosine"
} = {}) => {
  const operator = { cosine: "<=>", l2: "<->", ip: "<#>" }[metric] ?? "<=>";
  const toVector = values => `[${Array.from(values).join(",")}]`;

  return {
    async upsert(records) {
      const sql =
        `INSERT INTO ${table} (${idColumn}, ${vectorColumn}, ${metadataColumn}) ` +
        `VALUES ($1, $2, $3) ` +
        `ON CONFLICT (${idColumn}) DO UPDATE SET ` +
        `${vectorColumn} = EXCLUDED.${vectorColumn}, ` +
        `${metadataColumn} = EXCLUDED.${metadataColumn}`;

      for (const r of records) {
        await pool.query(sql, [r.id, toVector(r.values), r.metadata ?? {}]);
      }
    },

    async query({ vector, topK = 10 }) {
      const sql =
        `SELECT ${idColumn} AS id, ${metadataColumn} AS metadata, ` +
        `${vectorColumn} ${operator} $1 AS score ` +
        `FROM ${table} ORDER BY ${vectorColumn} ${operator} $1 LIMIT $2`;

      const { rows } = await pool.query(sql, [toVector(vector), topK]);

      // <=> (cosine) and <-> (L2) are distances (lower = closer); <#> is the
      // negative inner product.
      return {
        matches: rows.map(row => ({ id: row.id, score: row.score, metadata: row.metadata ?? {} }))
      };
    },

    async deleteMany(ids) {
      await pool.query(`DELETE FROM ${table} WHERE ${idColumn} = ANY($1)`, [ids]);
    }
  };
};

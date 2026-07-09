/**
 * vectorish — a tiny brute-force vector database.
 *
 * Implements the same Pinecone-style interface Embassy expects (see
 * ./vectordb.js) and answers queries with a linear-scan cosine similarity
 * search. No ANN index — meant for tests, local development, small datasets,
 * and examples.
 *
 * By default records live in an in-memory Map. Pass an optional `backing` to
 * store them somewhere else; the backing can be any of:
 *   - a Map-like object   ({ get, set, delete, values })
 *   - Web Storage         (localStorage / sessionStorage)
 *   - the Cache API        (a Cache from `await caches.open(name)`)
 *   - the CookieStore API  (globalThis.cookieStore)
 *   - IndexedDB            (the `indexedDB` factory or an open IDBDatabase)
 *
 *   import { Embassy } from "../embassy.js";
 *   import { vectorish } from "./adapters/vectorish.js";
 *
 *   const embassy = new Embassy(vectorish(), { ... });           // in-memory
 *   const persisted = new Embassy(vectorish(localStorage), {});  // survives reloads
 *
 * Records are stored as { id, values, metadata }. String-only backings (Web
 * Storage, CookieStore) hold a JSON encoding; the Cache API holds a JSON
 * Response. Every query scans all records, so keep datasets modest.
 */

// Cosine similarity over the overlapping prefix of the two vectors. Returns 0
// when either vector has zero magnitude (avoids divide-by-zero / NaN).
const cosine = (a, b) => {
  let dot = 0,
    na = 0,
    nb = 0;
  const len = Math.min(a.length, b.length);

  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }

  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
};

// Keep only records whose metadata matches every key in `filter` (shallow
// equality). No filter means every record qualifies.
const matchesFilter = (metadata, filter) => {
  if (!filter) return true;
  return Object.entries(filter).every(([key, value]) => metadata?.[key] === value);
};

// ---------------------------------------------------------------------------
// Backing adapters
//
// Each adapter normalizes a storage shape to the same minimal interface
// vectorish needs: set(id, record), delete(id) -> boolean, and values() -> an
// async iterable of the stored records. Records are plain objects of the form
// { id, values, metadata }.
// ---------------------------------------------------------------------------

// Namespaces keys in shared, string-keyed stores (Web Storage, CookieStore) so
// vectorish records don't collide with unrelated entries.
const KEY_PREFIX = "vectorish:";

// Map-like: a Map, or anything exposing get/set/delete plus a values() iterator.
const mapBacking = (map) => ({
  async set(id, record) {
    map.set(id, record);
  },
  async delete(id) {
    return map.delete(id);
  },
  async * values() {
    yield* map.values();
  }
});

// Web Storage (localStorage / sessionStorage): values must be strings, so
// records are JSON-encoded. Iteration walks the key(index) list.
const storageBacking = (storage) => ({
  async set(id, record) {
    storage.setItem(KEY_PREFIX + id, JSON.stringify(record));
  },
  async delete(id) {
    const key = KEY_PREFIX + id;
    const existed = storage.getItem(key) != null;
    storage.removeItem(key);
    return existed;
  },
  async * values() {
    for (let i = 0; i < storage.length; i++) {
      const key = storage.key(i);
      if (key?.startsWith(KEY_PREFIX)) {
        const raw = storage.getItem(key);
        if (raw != null) yield JSON.parse(raw);
      }
    }
  }
});

// Cache API: each record is stored as a JSON Response under a synthetic URL.
const CACHE_ORIGIN = "https://vectorish.invalid/";
const cacheUrl = id => CACHE_ORIGIN + encodeURIComponent(id);

const cacheBacking = (cache) => ({
  async set(id, record) {
    await cache.put(
      cacheUrl(id),
      new Response(JSON.stringify(record), {
        headers: {
          "content-type": "application/json"
        }
      })
    );
  },
  async delete(id) {
    return cache.delete(cacheUrl(id));
  },
  async * values() {
    for (const request of await cache.keys()) {
      const response = await cache.match(request);
      if (response) yield await response.json();
    }
  }
});

// CookieStore API: cookie values are strings, so records are URL-encoded JSON.
// Cookies are small (~4 KB each) — practical only for tiny vectors.
const cookieBacking = (cookieStore) => ({
  async set(id, record) {
    await cookieStore.set(KEY_PREFIX + id, encodeURIComponent(JSON.stringify(record)));
  },
  async delete(id) {
    const name = KEY_PREFIX + id;
    const existed = (await cookieStore.get(name)) != null;
    await cookieStore.delete(name);
    return existed;
  },
  async * values() {
    for (const {
        name,
        value
      }
      of await cookieStore.getAll()) {
      if (name.startsWith(KEY_PREFIX)) yield JSON.parse(decodeURIComponent(value));
    }
  }
});

// IndexedDB: accepts the `indexedDB` factory (opens its own DB/store) or an
// already-open IDBDatabase. Records use an in-line "id" key path.
const IDB_DB_NAME = "vectorish";
const IDB_STORE_NAME = "records";

const idbRequest = request => new Promise((resolve, reject) => {
  request.onsuccess = () => resolve(request.result);
  request.onerror = () => reject(request.error);
});

const idbBacking = (backing) => {
  let ready;

  // Resolve to { db, storeName }, opening a database when given a factory.
  const connect = () => {
    if (ready) return ready;

    if (typeof backing.open === "function") {
      ready = new Promise((resolve, reject) => {
        const request = backing.open(IDB_DB_NAME, 1);
        request.onupgradeneeded = () => {
          const db = request.result;
          if (!db.objectStoreNames.contains(IDB_STORE_NAME))
            db.createObjectStore(IDB_STORE_NAME, {
              keyPath: "id"
            });
        };
        request.onsuccess = () => resolve({
          db: request.result,
          storeName: IDB_STORE_NAME
        });
        request.onerror = () => reject(request.error);
      });
    } else {
      const storeName = backing.objectStoreNames?.contains(IDB_STORE_NAME) ?
        IDB_STORE_NAME :
        backing.objectStoreNames?.[0];
      ready = Promise.resolve({
        db: backing,
        storeName
      });
    }

    return ready;
  };

  const objectStore = async (mode) => {
    const {
      db,
      storeName
    } = await connect();
    return db.transaction(storeName, mode).objectStore(storeName);
  };

  return {
    async set(id, record) {
      const store = await objectStore("readwrite");
      await idbRequest(store.put(record));
    },
    async delete(id) {
      const store = await objectStore("readwrite");
      const existing = await idbRequest(store.get(id));
      await idbRequest(store.delete(id));
      return existing !== undefined;
    },
    async * values() {
      const store = await objectStore("readonly");
      yield* await idbRequest(store.getAll());
    }
  };
};

// Pick the adapter matching the backing's shape. Checks run from the most
// specific signature to the most general (Map-like) to avoid false matches.
const normalizeBacking = (backing) => {
  if (typeof backing.getItem === "function" && typeof backing.setItem === "function")
    return storageBacking(backing);

  if (typeof backing.match === "function" && typeof backing.put === "function")
    return cacheBacking(backing);

  if (typeof backing.getAll === "function" && typeof backing.set === "function")
    return cookieBacking(backing);

  // `cmp` marks an IDBFactory, `transaction` an IDBDatabase — neither exists on
  // a CacheStorage, so we don't confuse `caches` with `indexedDB`.
  if (typeof backing.cmp === "function" || typeof backing.transaction === "function")
    return idbBacking(backing);

  if (typeof backing.get === "function" &&
    typeof backing.set === "function" &&
    typeof backing.delete === "function")
    return mapBacking(backing);

  throw new TypeError(
    "vectorish: unsupported backing — expected a Map-like object, Web Storage, " +
    "Cache, CookieStore, or IndexedDB."
  );
};

/**
 * Create a vectorish database.
 *
 * @param {object} [backing] Optional persistence backing (Map-like, Web
 *   Storage, Cache, CookieStore, or IndexedDB). Defaults to an in-memory Map.
 *
 * The raw backing is exposed as `store` so callers can inspect or seed it
 * directly (for the default it is the in-memory Map).
 */
export const vectorish = (backing) => {
  const source = backing ?? new Map();
  const backend = normalizeBacking(source);

  return {
    store: source,

    async upsert(records) {
      for (const r of records) {
        await backend.set(r.id, {
          id: r.id,
          values: Array.from(r.values),
          metadata: r.metadata ?? {}
        });
      }
      return {
        upsertedCount: records.length
      };
    },

    async query({
      vector,
      topK = 10,
      includeMetadata = false,
      includeValues = false,
      filter
    } = {}) {
      const matches = [];

      for await (const rec of backend.values()) {
        if (!matchesFilter(rec.metadata, filter)) continue;

        matches.push({
          id: rec.id,
          score: cosine(vector, rec.values),
          ...(includeMetadata ? {
            metadata: rec.metadata
          } : {}),
          ...(includeValues ? {
            values: rec.values
          } : {})
        });
      }

      matches.sort((a, b) => b.score - a.score);
      return {
        matches: matches.slice(0, topK)
      };
    },

    async deleteMany(ids) {
      let deletedCount = 0;
      for (const id of ids) {
        if (await backend.delete(id)) deletedCount++;
      }
      return {
        deletedCount
      };
    }
  };
};

export default vectorish;

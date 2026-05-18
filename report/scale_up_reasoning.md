# Scale-Up Reasoning Report
## Scale-Up Reasoning - CineGraph BI: VFX Success Intelligence

---

## 1. The 5 Vs at Production Scale

Our current pipeline operates on a static TMDB snapshot. The table below contrasts what we actually built against what a production deployment would face.

| V | PoC (What We Built) | Production Scale (What It Becomes) |
|---|---------------------|-------------------------------------|
| **Volume** | 1.18M raw rows; 209,244 films in ML feature matrix; 202,164 movies embedded; 208,328 Qdrant vectors | Full TMDB catalog updated daily (~600K titles); add streaming platform metadata (Netflix, Prime, Disney+), user ratings (100M+ rows), daily box office per cinema (~10B rows/year) → total data lake exceeds 5TB |
| **Velocity** | Static weekly/monthly TMDB snapshot ingested in a single batch run | Daily delta ingestion of new releases, rating updates, box office figures; near-real-time crew announcements and casting news; Qdrant re-index triggered on each new embedding batch |
| **Variety** | Structured numerics (budget, runtime); semi-structured comma lists (genres, cast); free text (overview, tagline); 384-dim float vectors | Adds video trailer metadata, poster images, subtitle files, Twitter/Reddit sentiment streams, multi-language overviews, nested JSON from streaming APIs - shifts from pure SQL to multi-modal pipelines |
| **Veracity** | 30–40% of rows missing budget/revenue; 0-values cast to NULL; director nulls scrubbed; 38:1 class imbalance (203,923 flops vs 5,321 hits) | Crowdsourced rating drift over time; conflicting financial figures across sources (TMDB vs Box Office Mojo vs studio reports); casting data lags actual shoot dates; requires automated anomaly detection and source-of-truth reconciliation |
| **Value** | PageRank features contribute 44.5% of ML model importance; ROC-AUC 0.8973; concept-to-film search in <1 second | Greenlight scoring API consumed by studio execs in real time; talent shortlist tool replacing ad-hoc casting; daily ROI benchmarks per genre/director cluster; semantic search with constraint filters (rating ≥ 7, post-2015, budget < $50M) |

---

## 2. Pipeline Limits and Component Replacements

### DuckDB (ETL - NB01)

**When it breaks:** DuckDB scans CSV/Parquet from disk efficiently but builds intermediate aggregations in RAM. On a 16 GB laptop, DuckDB starts struggling around **8–10 GB of raw input** - roughly 5× our current 1.18M-row dataset. At that point, multi-table joins on unnested cast lists (which produce millions of intermediate rows) exhaust available memory and spill to disk, causing 10–20× slowdowns.

**Replacement:** Apache Spark (PySpark) running on a 3-node cluster. Spark's lazy evaluation and distributed shuffle handle hundreds of GB without memory pressure. The trade-off is real: Spark requires cluster provisioning (AWS EMR, Databricks, or local Docker Compose), adds ~30 minutes of startup overhead per job, and demands knowledge of partitioning strategy to avoid shuffle bottlenecks. For a team that only re-runs ETL weekly, that overhead is acceptable. For daily delta runs under 5 GB, DuckDB with incremental Parquet appends remains the right tool - Spark would be over-engineering.

**Honest limitation:** Our current DuckDB pipeline has no orchestration. A production system needs Apache Airflow (or Prefect) to schedule, retry, and alert on failed ingestion steps. We have not built this.

---

### Pandas (Feature Engineering - NB04)

**When it breaks:** Pandas loads the entire DataFrame into RAM. Our feature matrix (209,244 × 15, ~25 MB) is trivially small today. The break point is around **~5 GB in memory** - roughly 20M rows at our current feature width, which corresponds to adding full user-rating history per film.

**Replacement:** Polars. It uses lazy evaluation (query planning before execution), processes data in chunks, and runs 5–10× faster than Pandas on the same hardware due to its Rust backend. The migration cost from Pandas to Polars is low - the API is similar - making it the first upgrade to make, not Spark. Spark should only replace Polars if the data must be distributed across multiple machines (>100 GB).

---

### Neo4j Community Edition (Graph - NB02 & NB03)

**When it breaks:** Neo4j Community stores the entire graph in a single JVM process. Our graph has ~797K relationships across ~500K nodes and comfortably fits in 4 GB of heap. Neo4j Community starts degrading around **50–100M relationships** - at that scale, GDS algorithms like PageRank require projecting the full graph into memory, which exceeds available heap on a single machine.

**Replacement options and trade-offs:**

| Option | When to Use | Trade-Off |
|--------|------------|-----------|
| **Neo4j AuraDS (cloud)** | Up to ~1B relationships; managed service | ~$500–2,000/month; data leaves your infrastructure; vendor lock-in |
| **Neo4j Enterprise (self-hosted)** | Horizontal read replicas; causal clustering | Expensive licensing (~$50K+/year); requires dedicated DBA |
| **TigerGraph** | Native MPP graph; handles 10B+ edges | Steep learning curve; proprietary query language (GSQL); hard to migrate from Cypher |
| **JanusGraph + Cassandra** | Open-source; integrates with existing Cassandra/HBase clusters | Complex ops; slower traversal than Neo4j for hop-heavy queries; requires Gremlin expertise |

**Our honest recommendation:** For a VFX intelligence platform growing to 5–10M relationships (adding streaming platforms), **Neo4j AuraDS** is the pragmatic next step - same Cypher queries, no infrastructure management, GDS algorithms available. JanusGraph is only justified if the organisation already runs Cassandra and has the ops maturity to manage it.

---

### GDS PageRank + Louvain (NB03)

**When it breaks:** GDS requires projecting the named graph entirely into JVM heap. Our 797K-relationship projection worked cleanly in ~2 GB heap. At **~100M relationships**, a 16 GB heap is insufficient for in-memory projection - GDS will fail with `OutOfMemoryError`.

**Replacement:** GraphX (Apache Spark's graph library) supports distributed PageRank across a cluster. However, GraphX is significantly harder to use than GDS Cypher and requires moving graph data out of Neo4j into Spark RDDs. A more practical alternative for moderate scale (up to 500M edges) is **Neo4j GDS Enterprise**, which adds disk-backed projections that avoid heap limits. The trade-off is cost vs operational simplicity.

---

### scikit-learn Random Forest (ML - NB04)

**When it breaks:** scikit-learn trains in a single process. Our 209,244 × 15 matrix trains in under 2 minutes. At **~10M rows**, training time exceeds 30 minutes and RAM usage can exceed 32 GB for 100-tree forests with deep trees. Beyond that, single-node training becomes impractical.

**Replacement:** XGBoost distributed (via Dask or Spark) or Spark MLlib's Random Forest implementation. XGBoost distributed is the preferred first step - it uses the same tree-based approach, has a familiar API, and can be run on a single multi-core machine with `n_jobs=-1` before needing a cluster. Spark MLlib should only be introduced when training data is genuinely distributed across a data lake.

**Honest limitation:** Our model has a severe class imbalance problem (38:1) that we addressed only through ROC-AUC evaluation, not by applying SMOTE, class weighting, or threshold tuning. Hit recall of 0.08–0.09 is not production-grade. A production greenlight scoring API would require SMOTE oversampling on the training set and a tuned classification threshold to improve recall meaningfully before deployment.

---

### all-MiniLM-L6-v2 Embedding Model (NB05)

**When it breaks:** Not a scale question - the model runs per-batch regardless of dataset size. The issue is **quality**, not throughput. all-MiniLM-L6-v2 is English-optimised. Our TMDB data includes multilingual overviews and the model produces degraded embeddings for non-English text. Encoding 202,164 movies took 26 minutes on CPU - at 1M movies this becomes ~2 hours, which is acceptable for a weekly batch but not for daily updates.

**Replacement:** `paraphrase-multilingual-mpnet-base-v2` for multilingual datasets (same SentenceTransformers library, 50+ languages). For highest quality on English-only data, OpenAI `text-embedding-3-large` (3072-dim) via API costs approximately $0.13 per 1M tokens - encoding our 202K movies would cost ~$2–3 total, a trivial one-time cost that delivers meaningfully better semantic precision. The trade-off: API dependency, data leaving your infrastructure, per-call billing at production refresh rates.

---

### Qdrant Local Docker (Vector Search - NB05)

**When it breaks:** Our local Qdrant instance holds 208,328 vectors at 384 dimensions (~300 MB on disk). Qdrant handles this in RAM with millisecond latency. At **~5M vectors** (adding user-generated content, review embeddings, trailer scene embeddings), a single Docker container on a laptop exceeds available RAM and query latency climbs above 100ms - unusable for a live search API.

**Replacement:** Qdrant Cloud (managed) scales horizontally with automatic sharding. At 5M vectors, a 2-node Qdrant Cloud deployment costs approximately $200–400/month. The alternative is Pinecone (fully managed, simpler ops, higher cost) or Weaviate (open-source, supports hybrid keyword + vector search, more complex deployment). For this domain, **Qdrant Cloud is the natural migration** - same client API, no code changes required, just swap the connection string.

**What changes at millions of embeddings:** Approximate Nearest Neighbour (ANN) index construction time becomes significant - Qdrant's HNSW index at 10M vectors takes ~2–3 hours to build. Incremental indexing (adding new embeddings without full rebuild) must be part of the pipeline design, not an afterthought.

---

## 3. Batch vs. Stream: Where Real-Time Processing Adds Value

Our entire current pipeline is batch-oriented. The table below identifies which components would benefit from streaming and which should stay batch.

| Component | Current | Should It Stream? | Streaming Architecture |
|-----------|---------|-------------------|------------------------|
| TMDB ingestion | Weekly batch | No - TMDB updates weekly | Keep as scheduled Airflow DAG |
| Box office data | Not implemented | **Yes** - daily grosses drive time-sensitive greenlighting | Apache Kafka topic consuming Box Office Mojo API; Flink job aggregating daily grosses into Neo4j |
| Social sentiment | Not implemented | **Yes** - Twitter/Reddit buzz spikes within hours of trailer drops | Kafka + Spark Structured Streaming; sentiment scored per film per hour; fed into Qdrant payload |
| Casting announcements | Not implemented | **Yes** - crew changes affect PageRank scores immediately | Event-driven Neo4j MERGE triggered by news API webhook; GDS PageRank incremental update |
| ML re-training | Batch | No - model retrained weekly/monthly on full history | Airflow-scheduled batch job; no streaming benefit |
| Vector re-indexing | Weekly batch | No - 26 min CPU encoding not suitable for real-time | Nightly batch embedding of new/updated films; incremental Qdrant upsert |

**Honest assessment:** Streaming infrastructure (Kafka + Flink/Spark Streaming) adds significant operational complexity - dedicated cluster, schema registry, consumer group management, exactly-once semantics. For a team of 1–3 engineers building an internal studio tool, streaming is only justified for **social sentiment and box office feeds**, where latency genuinely affects decisions. Everything else runs correctly and cheaply as nightly Airflow DAGs.

---

## 4. Summary: Scale Thresholds at a Glance

| Component | Current Scale | Breaks At | Replace With | Why |
|-----------|--------------|-----------|--------------|-----|
| DuckDB | 1.18M rows, ~500 MB | ~8–10 GB RAM | PySpark on EMR / Databricks | Distributed shuffle handles multi-table joins beyond single-node RAM |
| Pandas | 209K × 15, ~25 MB | ~5 GB in memory | Polars first, then Spark | Polars lazy evaluation avoids full in-memory load; same API, lower friction |
| Neo4j Community | 797K relationships | ~50–100M relationships | Neo4j AuraDS | Same Cypher queries, managed scaling, GDS available; no code rewrite |
| Neo4j GDS | 797K-edge in-memory projection | ~100M edges in JVM heap | GDS Enterprise (disk-backed) | Avoids OOM on large projections without leaving the Neo4j ecosystem |
| scikit-learn RF | 209K × 15 | ~10M rows, ~32 GB RAM | XGBoost distributed, then Spark MLlib | Familiar API; multi-node training only when single-machine memory is exhausted |
| all-MiniLM-L6-v2 | 202K docs, 26 min CPU | Quality degrades on multilingual content | multilingual-mpnet (free) or text-embedding-3-large (API) | Multilingual model handles non-English overviews; OpenAI model gives best precision at negligible one-time cost |
| Qdrant local Docker | 208K vectors, ~300 MB RAM | ~5M vectors exceeds laptop RAM | Qdrant Cloud (2-node) | Same client API; automatic sharding; no code changes on migration |

---

*Scale thresholds are based on empirical benchmarks from tool documentation and community benchmarks, cross-referenced against observed runtime and memory usage during development of this pipeline. This section was drafted after NB01–NB02 and finalised after NB05 once the full pipeline was operational.*

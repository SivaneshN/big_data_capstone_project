# Scale-Up Reasoning Report
## VFX Movies Neo4j Graph Pipeline

---

## 1. Tool Comparison Table

| Tool | Role in Pipeline | Strength | Limitation | Scale-Up Alternative |
|------|-----------------|----------|------------|----------------------|
| **DuckDB** | In-process SQL engine for ETL & cleaning | Zero-server setup; blazing fast analytical queries on local CSV/Parquet | Single-node; no concurrent writes; not suited for >100 GB in-memory workloads | Apache Spark / Trino for distributed SQL at petabyte scale |
| **Pandas** | DataFrame manipulation & CSV I/O | Intuitive API; seamless DuckDB integration | Loads entire dataset into RAM; struggles beyond ~5 GB | Polars (Rust-backed, lazy evaluation) or Dask for out-of-core processing |
| **Neo4j (Community)** | Graph storage, traversal, Cypher queries | Native graph model; millisecond traversal for connected data | Single-instance; no horizontal sharding in Community edition | Neo4j AuraDS (cloud) or TigerGraph for multi-shard graph clusters |
| **Python neo4j driver** | Batch-load CSVs into Neo4j via Bolt | Flexible; supports MERGE + constraints; easy batching | Serial batch loop; ~1 000 rows/batch is slow for millions of edges | Neo4j `LOAD CSV` bulk import or `neo4j-admin import` for 10M+ relationships |
| **Parquet (Snappy)** | Persistent snapshot after cleaning | Columnar compression; 3–5× smaller than CSV; query-ready by Spark/DuckDB | Schema drift if upstream CSV columns change | Apache Iceberg / Delta Lake for ACID-compliant, versioned data lakes |

---

## 2. The 5 Vs of the Dataset

| V | Description | In Our Dataset |
|---|-------------|----------------|
| **Volume** | Total size of data at rest | ~500 K+ movie records in `TMDB_all_movies.csv`; ~180 K VFX-filtered rows; multi-million actor–movie edges after unnesting cast lists |
| **Velocity** | Speed at which data arrives or must be processed | TMDB snapshots released periodically (weekly/monthly); pipeline must re-ingest and re-merge without duplicating nodes — handled via `MERGE` + unique constraints |
| **Variety** | Diversity of data types and structures | Structured numerics (budget, revenue, runtime); semi-structured lists (genres, cast, production companies as comma-separated strings); free text (overview, tagline) |
| **Veracity** | Trustworthiness and quality of data | ~30–40 % of rows lack budget/revenue; director nulls scrubbed; `TRY_CAST` guards against malformed dates; 0-values treated as `NULL` for financial fields |
| **Value** | Business utility extracted from the data | Cypher queries surface top VFX actors, director–actor collaboration patterns, and cross-genre actor versatility — directly actionable for casting and acquisition decisions |

---

## 3. Why a Graph Model Adds Value Over Relational SQL

A relational approach requires expensive multi-table JOINs to answer questions like *"which actors worked with 3+ directors across genres?"*. Neo4j stores these as direct pointer traversals — O(log n) lookups regardless of graph size — making relationship-heavy queries orders of magnitude faster at scale.

The `(Actor)-[:ACTED_IN]->(Movie)<-[:DIRECTED]-(Director)` pattern is a natural fit for graph; in SQL it would require a three-way JOIN across potentially millions of rows.

---

*Report generated as part of the VFX Movies Big Data capstone project.*

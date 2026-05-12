# Mid-Term Presentation Outline
## VFX Movies Neo4j Graph Pipeline
### 5 Parts

---

## Part 1 — What We Have

- **Dataset:** `TMDB_all_movies.csv` — 500 K+ movie records sourced from The Movie Database
- **ETL Notebook (`notebooks/01_etl.ipynb`):**
  - Loaded raw CSV into DuckDB in-process engine
  - Cleaned nulls, cast types, computed `roi_pct`
  - Filtered ~180 K VFX-relevant movies using genre + keyword signals
  - Exported `vfx_movies_1.parquet` and five Neo4j-ready CSVs
- **Graph Schema:**
  - Nodes: `Movie`, `Director`, `Actor`
  - Relationships: `(Director)-[:DIRECTED]->(Movie)`, `(Actor)-[:ACTED_IN]->(Movie)`
  - Constraints: unique `movieId`, unique director/actor `name`
- **Post-Load Counts:**
  - Movie nodes: ~180 K
  - Director nodes: ~55 K
  - Actor nodes: ~420 K
  - DIRECTED edges: ~180 K
  - ACTED_IN edges: ~2.1 M

---

## Part 2 — Live Demo: 3 Cypher Queries

**Query 1 — Most Prolific Actors in VFX Movies**
```cypher
MATCH (a:Actor)-[:ACTED_IN]->(m:Movie)
RETURN a.name AS actor, COUNT(m) AS movies_count
ORDER BY movies_count DESC LIMIT 10
```
*Business value: Identifies bankable VFX talent for casting decisions.*

**Query 2 — Director–Actor Collaborations (2+ shared films)**
```cypher
MATCH (d:Director)-[:DIRECTED]->(m:Movie)<-[:ACTED_IN]-(a:Actor)
WITH d.name AS director, a.name AS actor, COUNT(m) AS collaborations
WHERE collaborations >= 2
RETURN director, actor, collaborations
ORDER BY collaborations DESC LIMIT 15
```
*Business value: Maps trusted creative partnerships for greenlight decisions.*

**Query 3 — Actors Who Worked with 3+ Directors Across Genres**
```cypher
MATCH (a:Actor)-[:ACTED_IN]->(m:Movie)<-[:DIRECTED]-(d:Director)
WHERE m.genres IS NOT NULL
WITH a.name AS actor, COUNT(DISTINCT d.name) AS directors_worked_with,
     COUNT(DISTINCT m.genres) AS genre_spread
WHERE directors_worked_with >= 3
RETURN actor, directors_worked_with, genre_spread
ORDER BY directors_worked_with DESC LIMIT 10
```
*Business value: Surfaces versatile actors with broad industry reach.*

---

## Part 3 — Business Question + Decision-Maker

**Business Question:**  
*"Which actors and directors should a studio prioritize for VFX-heavy productions?"*

**Decision-Maker:** Streaming Platform Analyst

**How the graph helps:**
- Query 1 → shortlist high-volume VFX actors proven to drive audience engagement
- Query 2 → identify director–actor pairs with established chemistry to reduce production risk
- Query 3 → find genre-flexible actors who can anchor diverse VFX slates on the platform

**Without a graph:** These questions require multi-table SQL JOINs across cast, crew, and movie tables — slow, brittle, and hard to extend. Neo4j answers them in milliseconds and the model extends naturally (add `Studio`, `Franchise`, `Award` nodes with zero schema migration).

---

## Part 4 — Product Vision: Unstructured Text

**The New Piece:** Semantic similarity search over `Movie.plot` (overview field)

**Text field we embed:** `Movie.overview` — free-text plot descriptions (~100–300 words per film)

**Embedding model:** `sentence-transformers` (`all-MiniLM-L6-v2`) → 384-dimensional vectors

**Streamlit Widget — User Flow:**
1. Curator pastes a free-text description of a film concept they want to acquire
2. App computes the embedding of the input text
3. Cosine similarity is computed against all stored `Movie.overview` embeddings
4. Returns **5 nearest films** ranked by similarity score

**User Journey:**
> A curator preparing an acquisition slate describes an unmade film concept.  
> The widget instantly surfaces the 5 most similar existing VFX films.  
> The curator sees adjacent titles for **pricing reference** and competitive positioning.  
> They attach the output directly to their acquisition proposal.

**Technical Stack Addition:**
- `sentence-transformers` for embedding generation
- `numpy` / `faiss` for fast cosine similarity at scale
- `Streamlit` for the no-code curator interface
- Embeddings stored as a property on `Movie` nodes in Neo4j (vector index via Neo4j 5.x)

---

## Part 5 — Gap Analysis + Next Steps

**What's Missing:**
| Gap | Impact | Mitigation |
|-----|--------|-----------|
| `budget` / `revenue` null in ~35 % of rows | ROI queries incomplete | Enrich via TMDB API or IMDb supplement |
| No `keywords` column in CSV | VFX filter relies on genre + text heuristics; some false positives | Add keyword-based filter once enriched |
| Embeddings not yet generated | Product vision is a sketch | Sprint 2 task: run embedding script offline, store vectors |
| Neo4j Community — single node | No HA, no sharding | Move to Neo4j AuraDS for production |
| No automated re-ingestion | Pipeline is manual | Wrap in Airflow DAG or GitHub Actions cron |

**Next Steps:**
1. Generate `overview` embeddings with `sentence-transformers` and store in Parquet
2. Build Streamlit similarity widget (local demo)
3. Enrich budget/revenue nulls via TMDB API calls
4. Add `Studio` and `Genre` as first-class nodes to enable richer traversals

**Risks:**
- TMDB data licensing — confirm terms before any production deployment
- Embedding quality degrades for very short or missing plot summaries
- Graph size may outpace Neo4j Community memory limits beyond 10 M edges

---

*Product Vision is the new piece — sketch, not commitment.*

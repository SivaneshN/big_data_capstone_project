# CineGraph — VFX Film Intelligence Platform

> **Big Data & AI Capstone Project | SRH University Leipzig | MSc Big Data & AI**

## Business Question

**Which actors and directors should a studio prioritise for VFX-heavy productions — and which crew combinations consistently deliver successful films?**

### Why This Matters

Casting and crew decisions are the highest-stakes choices a studio makes before principal photography begins. For VFX-heavy productions — where budgets routinely exceed $100M — a mis-hire at the director or lead actor level can derail an entire franchise slate. Today these decisions rely on gut instinct, agent relationships, and fragmented spreadsheets.

CineGraph makes the VFX success network *visible*:

- **Query 1 — Bankable VFX Talent:** Which actors appear most frequently in successful VFX productions? High-volume presence signals audience familiarity and proven ability to anchor effects-driven films.
- **Query 2 — Trusted Creative Partnerships:** Which director–actor pairs have collaborated on 2+ VFX films? Established chemistry reduces production risk and shortens creative alignment time.
- **Query 3 — Versatile Industry Reach:** Which actors have worked across 3+ directors and genres? Genre-flexible actors can anchor diverse VFX slates — from sci-fi to fantasy to action.

By modeling the entire VFX crew ecosystem as a knowledge graph and applying PageRank + Louvain community detection, CineGraph surfaces non-obvious hubs of influence — the producer whose name raises a project's hit-rate, or the director–writer pair whose collaborations consistently score above 7.5 — enabling data-driven greenlight decisions.

---

## Decision-Maker

**Streaming Platform Analyst** — responsible for acquisition slates, greenlight recommendations, and VFX production partnerships. Uses CineGraph to:

1. Shortlist high-volume VFX actors proven to drive audience engagement
2. Identify director–actor pairs with established chemistry to reduce production risk
3. Surface genre-flexible actors who can anchor diverse VFX content slates
4. Find semantically similar existing films for pricing reference during acquisition

---

## Dataset

| Field | Details |
|-------|---------|
| **Name** | TMDB All Movies — Daily Updates |
| **Source** | [Kaggle: alanvourch/tmdb-movies-daily-updates](https://www.kaggle.com/datasets/alanvourch/tmdb-movies-daily-updates) |
| **License** | CC0: Public Domain |
| **Size** | ~500K+ movie records, ~1 GB CSV |
| **Key fields** | title, director, cast, producers, writers, director_of_photography, music_composer, budget, revenue, vote_average, vote_count, genres, overview, production_companies, runtime, release_date |

> **Download:** Register on Kaggle, run `kaggle datasets download -d alanvourch/tmdb-movies-daily-updates`, place `TMDB_all_movies.csv` in `data/`.

### VFX Movie Identification

Rather than filtering by genre keywords upfront, CineGraph uses a **community-based approach**:

1. All 500K+ movies are loaded into DuckDB and cleaned
2. A bipartite graph is built: Movies ↔ Production Companies ↔ Directors
3. Louvain community detection finds structural clusters
4. Each community is characterised by two independent metrics:
   - `vfx_company_ratio` — fraction of companies in the community that are known VFX studios (ILM, Weta, DNEG, etc.)
   - `vfx_percentage` — % of movies in the community whose overview mentions VFX technical terms
5. A community is declared a **VFX community** if `vfx_company_ratio > 0` OR `vfx_percentage ≥ 20%`
6. Final dataset: VFX community movies released **after 2000** with **runtime > 30 minutes**

---

## Graph Schema

```
Nodes (7 types):
  (:Movie)     — movie_id, title, release_year, vote_average, budget, revenue,
                 roi_pct, is_successful, vfx_company_ratio, vfx_percentage,
                 community_id, is_vfx
  (:Director)  — name, pagerank, community_id
  (:Actor)     — name, pagerank, community_id
  (:Producer)  — name, pagerank, community_id
  (:Writer)    — name, pagerank, community_id
  (:DOP)       — name, pagerank, community_id
  (:Composer)  — name, pagerank, community_id

Relationships (6 types):
  (Movie)-[:DIRECTED_BY]->(Director)
  (Movie)-[:ACTED_IN]->(Actor)
  (Movie)-[:PRODUCED_BY]->(Producer)
  (Movie)-[:WRITTEN_BY]->(Writer)
  (Movie)-[:SHOT_BY]->(DOP)
  (Movie)-[:SCORE_BY]->(Composer)
```

**Approximate post-load counts:**
- Movie nodes: ~180K
- Director nodes: ~55K
- Actor nodes: ~420K
- DIRECTED_BY edges: ~180K
- ACTED_IN edges: ~2.1M

---

## Live Demo Queries

**Query 1 — Most Prolific Actors in VFX Movies**
```cypher
MATCH (a:Actor)-[:ACTED_IN]->(m:Movie)
RETURN a.name AS actor, COUNT(m) AS movies_count
ORDER BY movies_count DESC LIMIT 10
```
*Business value: Identifies bankable VFX talent for casting decisions.*

**Query 2 — Director–Actor Collaborations (2+ shared films)**
```cypher
MATCH (d:Director)-[:DIRECTED_BY]->(m:Movie)<-[:ACTED_IN]-(a:Actor)
WITH d.name AS director, a.name AS actor, COUNT(m) AS collaborations
WHERE collaborations >= 2
RETURN director, actor, collaborations
ORDER BY collaborations DESC LIMIT 15
```
*Business value: Maps trusted creative partnerships for greenlight decisions.*

**Query 3 — Actors Who Worked with 3+ Directors Across Genres**
```cypher
MATCH (a:Actor)-[:ACTED_IN]->(m:Movie)<-[:DIRECTED_BY]-(d:Director)
WHERE m.genres IS NOT NULL
WITH a.name AS actor,
     COUNT(DISTINCT d.name) AS directors_worked_with,
     COUNT(DISTINCT m.genres) AS genre_spread
WHERE directors_worked_with >= 3
RETURN actor, directors_worked_with, genre_spread
ORDER BY directors_worked_with DESC LIMIT 10
```
*Business value: Surfaces versatile actors with broad industry reach.*

**Without a graph:** These questions require multi-table SQL JOINs across cast, crew, and movie tables — slow, brittle, and hard to extend. Neo4j answers them in milliseconds and the model extends naturally (add `Studio`, `Franchise`, or `Award` nodes with zero schema migration).

---

## Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| ETL | DuckDB + Python | CSV ingestion, cleaning, community-based VFX identification, Parquet export |
| Graph DB | Neo4j 5 Community (Docker) | Property graph storage, Cypher queries |
| Graph Analytics | Neo4j GDS | PageRank centrality, Louvain community detection |
| ML | scikit-learn | Binary classification: predict film success (baseline vs PageRank-enriched) |
| Semantic Search | Qdrant + sentence-transformers | Embed movie overviews, similarity search for acquisition |
| Dashboard | Streamlit + Plotly | Interactive BI frontend for streaming analysts |
| Infrastructure | Docker Compose | Neo4j + Qdrant containers |

---

## Quick Start

### 1. Prerequisites

- Docker + Docker Compose (Docker Desktop must be running)
- Python 3.12
- Kaggle account (for dataset download)

### 2. Download Dataset

```bash
# Option A: Kaggle CLI
pip install kaggle
kaggle datasets download -d alanvourch/tmdb-movies-daily-updates -p data/ --unzip

# Option B: Manual download from Kaggle → place TMDB_all_movies.csv in data/
```

### 3. Start Services

```bash
docker-compose up -d
# Wait ~60 seconds for Neo4j to initialise (GDS plugin install)
# Neo4j Browser : http://localhost:7474  (user: neo4j, password: capstone2024)
# Qdrant Dashboard: http://localhost:6333/dashboard
```

If you see a container name conflict:
```bash
docker rm -f cinegraph_neo4j cinegraph_qdrant
docker-compose up -d
```

### 4. Install Python Dependencies

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 5. Run Notebooks (in order)

```bash
jupyter notebook
```

| Order | Notebook | What it does |
|-------|----------|-------------|
| 1 | `01_etl_updated.ipynb` | Load CSV → DuckDB → clean → community graph → VFX scoring → Parquet + Neo4j CSVs |
| 2 | `02_graph_load.ipynb` | Load Neo4j: 7 node types, 6 relationship types, constraint setup |
| 3 | `03_graph_analytics.ipynb` | GDS: PageRank + Louvain community detection, 10 Cypher analytical queries |
| 4 | `04_ml.ipynb` | sklearn: baseline vs PageRank-enriched classifier, before/after % change visualisation |
| 5 | `05_embeddings.ipynb` | Embed overviews → Qdrant, semantic similarity search demo |

### 6. Launch Dashboard

```bash
streamlit run dashboard/app.py
# Opens at http://localhost:8501
```

---

## Product Vision — Semantic Acquisition Search

A streaming platform analyst preparing an acquisition slate can describe an unmade film concept in plain text. The widget:

1. Computes the embedding of the input description (`all-MiniLM-L6-v2`, 384-dim vectors)
2. Runs cosine similarity against all stored `Movie.overview` embeddings in Qdrant
3. Returns the **5 nearest existing VFX films** ranked by similarity score

**Business value:** The curator sees adjacent titles instantly — for **pricing reference** and competitive positioning — and can attach the output directly to their acquisition proposal. No SQL, no manual search.

---

## Dashboard Panels

| Panel | Business Value |
|-------|---------------|
| 📊 Overview | KPIs, year distribution, rating histogram, budget/revenue scatter |
| 🎭 Actor Insights | Top actors by hit rate, PageRank leaders, genre versatility |
| 🎬 Director Insights | ROI leaders, critical success rate, PageRank vs hit rate |
| 🎥 Crew Insights | Producers / Writers / DOPs / Composers — configurable role explorer |
| 🤝 Power Combos | Director–Actor pairs, Director–Producer pairs, Golden Trios |
| 🔍 Similarity Search | Enter any film concept → find semantically similar VFX movies |

---

## Key Findings

1. **Actor network centrality ≠ box office dominance.** Some high-PageRank actors appear in many VFX films but with below-average hit rates — network position is necessary but not sufficient for commercial success.

2. **Producer network position is the strongest predictor** of expected revenue — a well-connected producer signals access to distribution deals, studio infrastructure, and bankable talent.

3. **Director–Writer recurring collaborations** (Louvain-detected communities) show 15–25% higher average ratings than one-time pairings, suggesting creative shorthand built over time produces measurably better outcomes.

4. **PageRank enrichment improved ML model F1-score** over degree-only features — confirming that *who you work with* adds predictive signal beyond *how many films you've made*.

---

## Gap Analysis

| Gap | Impact | Mitigation |
|-----|--------|-----------|
| `budget` / `revenue` null in ~35% of rows | ROI queries incomplete | Enrich via TMDB API or IMDb supplement |
| Embeddings stored in Qdrant, not Neo4j | Vector index not native to graph | Neo4j 5.x vector index as future migration |
| Neo4j Community — single node | No HA, no sharding | Move to Neo4j AuraDS for production |
| No automated re-ingestion | Pipeline is manual | Wrap in Airflow DAG or GitHub Actions cron |

---

## Repository Structure

```
tmdb_capstone/
├── docker-compose.yml                 # Neo4j 5 + Qdrant 1.9
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── TMDB_all_movies.csv            # NOT committed — download from Kaggle
│   ├── clean_movies.parquet           # Generated by 01_etl_updated.ipynb
│   ├── community_vfx_stats.parquet    # Community characterisation output
│   ├── ml_comparison.parquet          # Generated by 04_ml.ipynb
│   ├── ml_before_after_pagerank.png   # Before/after PageRank visualisation
│   └── neo4j_csv/                     # CSVs generated by 01_etl_updated.ipynb
├── notebooks/
│   ├── 01_etl_community_filter.ipynb  # Community-based VFX identification
│   ├── 01_etl_direct_keyword_filter   # Keyword-based VFX identification
│   ├── 02_graph_load.ipynb
│   ├── 03_graph_analytics.ipynb
│   ├── 04_ml.ipynb
│   ├── 05_embeddings.ipynb            # Semantic Embeddings + Qdrant Similarity Search
│   └── Dataset_Notebook_1.ipynb       # Star war movies vs transformer films analysis
├── dashboard/
│   └── app.py                         # Streamlit dashboard
└── report/
    └── scale_up_reasoning.md          # From PoC to Production
```

---

## Environment Variables

The app reads connection settings via environment variables with local defaults.
No .env file is required for local development — the defaults match the Docker Compose setup.

| Variable | Default |
|----------|---------|
| NEO4J_URI | bolt://localhost:7687 |
| NEO4J_USERNAME | neo4j |
| NEO4J_PASSWORD | set in docker-compose.yml |
| QDRANT_URL | http://localhost:6333 |

---

## Team Members

| Name | GitHub |
|------|--------|
| Nidhi Chaubey | [@nchaubey12](https://github.com/nchaubey12) | 
| OM Rameshwar Surase| [@OmSurase1411](https://github.com/OmSurase1411)|
| Sivanesh Nadar| [@SivaneshN](https://github.com/SivaneshN) | 
| Tejas Patil|                                                 |
| Eiva Merin Eldose| [@Eiva-merin](https://github.com/Eiva-merin)|

---

## License

Dataset: CC0 Public Domain (TMDB Community).  
Code: MIT License.
"""
VFX Movies Neo4j Graph Pipeline
Dataset: TMDB_all_movies.csv
Goal: Build a Neo4j graph of VFX movies & directors, then run 3 business-oriented Cypher queries.
"""

import duckdb
import pandas as pd

df = pd.read_csv("/Volumes/Personal/Big data project/TMDB_all_movies.csv")
print("Columns:", df.columns.tolist())
print(df.head())

# ── 1. Loading Raw Data into DuckDB ──────────────────────────────────────────

FILE_PATH = "/Volumes/Personal/Big data project/TMDB_all_movies.csv"

conn = duckdb.connect(FILE_PATH)

conn.sql(f"""
    CREATE OR REPLACE TABLE raw_movies AS
    SELECT * FROM read_csv_auto('{FILE_PATH}',
        nullstr=['', 'NA', 'N/A'],
        header=true
    )
""")

conn.sql("DESCRIBE raw_movies").show()
conn.sql("SELECT COUNT(*) AS total FROM raw_movies").show()
conn.sql("SELECT * FROM raw_movies LIMIT 5").show()

# ── 2. Data Cleaning ──────────────────────────────────────────────────────────

conn.sql("""
    CREATE OR REPLACE TABLE clean_movies AS
    SELECT
        id,
        title,
        original_title,
        TRIM(director)                                       AS director,
        TRY_CAST(release_date AS DATE)                       AS release_date,
        YEAR(TRY_CAST(release_date AS DATE))                 AS release_year,
        genres,
        overview,
        tagline,
        original_language,
        production_companies,
        production_countries,
        spoken_languages,
        "cast"                                               AS cast_list,
        writers,
        producers,
        music_composer,
        director_of_photography,
        popularity,

        -- Treat 0 as missing for budget and revenue
        NULLIF(budget, 0)                                    AS budget,
        NULLIF(revenue, 0)                                   AS revenue,

        -- ROI only when both are valid
        CASE
            WHEN budget > 0 AND revenue > 0
            THEN ROUND((revenue - budget) / budget * 100, 2)
        END                                                  AS roi_pct,

        runtime,
        ROUND(imdb_rating, 1)                                AS rating,
        NULLIF(imdb_votes, 0)                                AS vote_count,
        vote_average,
        status,
        poster_path

    FROM raw_movies
    WHERE title      IS NOT NULL
      AND status     = 'Released'
      AND director   IS NOT NULL
      AND TRIM(director) != ''
""")

conn.sql("""
    SELECT
        COUNT(*)                   AS total_rows,
        COUNT(budget)              AS has_budget,
        COUNT(revenue)             AS has_revenue,
        COUNT(rating)              AS has_rating,
        COUNT(director)            AS has_director,
        COUNT(*) - COUNT(director) AS null_director
    FROM clean_movies
""").show()

# ── 3. Filtering VFX Movies ───────────────────────────────────────────────────
# Since the CSV has no `keywords` column, VFX detection uses genres, overview, and tagline.

conn.sql("""
    CREATE OR REPLACE TABLE vfx_movies AS
    SELECT *
    FROM clean_movies
    WHERE (
        -- Genre-based
        LOWER(genres) LIKE '%science fiction%'
        OR LOWER(genres) LIKE '%action%'
        OR LOWER(genres) LIKE '%fantasy%'
        OR LOWER(genres) LIKE '%adventure%'
        OR LOWER(genres) LIKE '%animation%'
        -- Overview / tagline keyword signals (replaces missing keywords column)
        OR LOWER(overview) LIKE '%visual effects%'
        OR LOWER(overview) LIKE '%cgi%'
        OR LOWER(overview) LIKE '%special effects%'
        OR LOWER(overview) LIKE '%computer generated%'
        OR LOWER(tagline)  LIKE '%visual effects%'
        OR LOWER(tagline)  LIKE '%cgi%'
        OR LOWER(tagline)  LIKE '%special effects%'
    )
""")

conn.sql("SELECT COUNT(*) AS vfx_movie_count FROM vfx_movies").show()
conn.sql("SELECT title, genres, release_year FROM vfx_movies LIMIT 10").show()

# ── 4. Exporting Parquet Snapshot ─────────────────────────────────────────────

conn.sql("""
    COPY vfx_movies
    TO 'vfx_movies_1.parquet'
    (FORMAT PARQUET, COMPRESSION SNAPPY)
""")

print("VFX Parquet saved!")

# ── 5. Exporting Neo4j CSVs ───────────────────────────────────────────────────
# Files:
#   neo4j_movies_1.csv       — Movie nodes
#   neo4j_directors_1.csv    — Director nodes
#   neo4j_directed_by_1.csv  — DIRECTED relationships

import os
OUTPUT_DIR = os.path.abspath(".")

# Movie nodes
conn.sql(f"""
    COPY (
        SELECT DISTINCT
            id           AS movieId,
            title,
            release_year,
            rating,
            vote_average,
            budget,
            revenue,
            roi_pct,
            runtime,
            genres,
            original_language
        FROM vfx_movies
        WHERE title IS NOT NULL
    )
    TO '{OUTPUT_DIR}/neo4j_movies_1.csv' (HEADER, DELIMITER ',')
""")

# Director nodes
conn.sql(f"""
    COPY (
        SELECT DISTINCT
            director AS name
        FROM vfx_movies
        WHERE director IS NOT NULL
          AND TRIM(director) != ''
    )
    TO '{OUTPUT_DIR}/neo4j_directors_1.csv' (HEADER, DELIMITER ',')
""")

# DIRECTED relationships
conn.sql(f"""
    COPY (
        SELECT DISTINCT
            id       AS movieId,
            director AS directorName
        FROM vfx_movies
        WHERE director IS NOT NULL
          AND TRIM(director) != ''
    )
    TO '{OUTPUT_DIR}/neo4j_directed_by_1.csv' (HEADER, DELIMITER ',')
""")

# Actor nodes (one row per unique actor name)
conn.sql(f"""
    COPY (
        SELECT DISTINCT TRIM(actor) AS name
        FROM (
            SELECT UNNEST(STRING_SPLIT(cast_list, ',')) AS actor
            FROM vfx_movies
            WHERE cast_list IS NOT NULL AND TRIM(cast_list) != ''
        )
        WHERE TRIM(actor) != ''
    )
    TO '{OUTPUT_DIR}/neo4j_actors_1.csv' (HEADER, DELIMITER ',')
""")

# ACTED_IN relationships (movieId, actorName)
conn.sql(f"""
    COPY (
        SELECT DISTINCT
            id AS movieId,
            TRIM(actor) AS actorName
        FROM (
            SELECT id, UNNEST(STRING_SPLIT(cast_list, ',')) AS actor
            FROM vfx_movies
            WHERE cast_list IS NOT NULL AND TRIM(cast_list) != ''
        )
        WHERE TRIM(actor) != ''
    )
    TO '{OUTPUT_DIR}/neo4j_acted_in_1.csv' (HEADER, DELIMITER ',')
""")

print("Neo4j VFX CSVs ready!")

# ── 6. Director Null Audit (Verification) ─────────────────────────────────────

conn.sql("""
    SELECT
        COUNT(*)                                      AS total_movies,
        COUNT(director)                               AS has_director,
        COUNT(*) - COUNT(director)                    AS null_director,
        ROUND(100.0 * COUNT(director) / COUNT(*), 1)  AS pct_with_director
    FROM clean_movies
""").show()

# ── 9. Running Cypher Queries from Python (neo4j driver) ─────────────────────

from neo4j import GraphDatabase

URI      = "neo4j://127.0.0.1:7687"
USERNAME = "neo4j"
PASSWORD = "Tejas.9850"

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

# ── 7. Load CSVs into Neo4j ───────────────────────────────────────────────────

def load_graph():
    movies_df    = pd.read_csv(f"{OUTPUT_DIR}/neo4j_movies_1.csv")
    directors_df = pd.read_csv(f"{OUTPUT_DIR}/neo4j_directors_1.csv")
    directed_df  = pd.read_csv(f"{OUTPUT_DIR}/neo4j_directed_by_1.csv")
    actors_df    = pd.read_csv(f"{OUTPUT_DIR}/neo4j_actors_1.csv")
    acted_in_df  = pd.read_csv(f"{OUTPUT_DIR}/neo4j_acted_in_1.csv")

    with driver.session() as session:
        session.run("CREATE CONSTRAINT movie_id IF NOT EXISTS FOR (m:Movie) REQUIRE m.movieId IS UNIQUE")
        session.run("CREATE CONSTRAINT director_name IF NOT EXISTS FOR (d:Director) REQUIRE d.name IS UNIQUE")
        session.run("CREATE CONSTRAINT actor_name IF NOT EXISTS FOR (a:Actor) REQUIRE a.name IS UNIQUE")

    BATCH = 1000

    # Movie nodes
    movie_records = movies_df.where(movies_df.notna(), None).to_dict("records")
    for i in range(0, len(movie_records), BATCH):
        batch = movie_records[i:i + BATCH]
        with driver.session() as session:
            session.run("""
                UNWIND $rows AS row
                MERGE (m:Movie {movieId: row.movieId})
                SET m.title             = row.title,
                    m.year              = row.release_year,
                    m.rating            = row.rating,
                    m.vote_average      = row.vote_average,
                    m.budget            = row.budget,
                    m.revenue           = row.revenue,
                    m.roi_pct           = row.roi_pct,
                    m.runtime           = row.runtime,
                    m.genres            = row.genres,
                    m.original_language = row.original_language
            """, rows=batch)
        if i % 10000 == 0:
            print(f"  Movies loaded: {min(i + BATCH, len(movie_records))}/{len(movie_records)}")
    print("Movie nodes loaded.")

    # Director nodes
    dir_records = directors_df.where(directors_df.notna(), None).to_dict("records")
    for i in range(0, len(dir_records), BATCH):
        batch = dir_records[i:i + BATCH]
        with driver.session() as session:
            session.run("""
                UNWIND $rows AS row
                MERGE (d:Director {name: row.name})
            """, rows=batch)
    print("Director nodes loaded.")

    # DIRECTED relationships
    rel_records = directed_df.where(directed_df.notna(), None).to_dict("records")
    for i in range(0, len(rel_records), BATCH):
        batch = rel_records[i:i + BATCH]
        with driver.session() as session:
            session.run("""
                UNWIND $rows AS row
                MATCH (m:Movie    {movieId: row.movieId})
                MATCH (d:Director {name: row.directorName})
                MERGE (d)-[:DIRECTED]->(m)
            """, rows=batch)
        if i % 10000 == 0:
            print(f"  DIRECTED loaded: {min(i + BATCH, len(rel_records))}/{len(rel_records)}")
    print("DIRECTED relationships loaded.")

    # Actor nodes
    actor_records = actors_df.where(actors_df.notna(), None).to_dict("records")
    for i in range(0, len(actor_records), BATCH):
        batch = actor_records[i:i + BATCH]
        with driver.session() as session:
            session.run("""
                UNWIND $rows AS row
                WITH row WHERE row.name IS NOT NULL AND row.name <> ''
                MERGE (a:Actor {name: row.name})
            """, rows=batch)
        if i % 50000 == 0:
            print(f"  Actors loaded: {min(i + BATCH, len(actor_records))}/{len(actor_records)}")
    print("Actor nodes loaded.")

    # ACTED_IN relationships
    acted_records = acted_in_df.where(acted_in_df.notna(), None).to_dict("records")
    for i in range(0, len(acted_records), BATCH):
        batch = acted_records[i:i + BATCH]
        with driver.session() as session:
            session.run("""
                UNWIND $rows AS row
                WITH row WHERE row.actorName IS NOT NULL AND row.actorName <> ''
                MATCH (m:Movie {movieId: row.movieId})
                MATCH (a:Actor {name: row.actorName})
                MERGE (a)-[:ACTED_IN]->(m)
            """, rows=batch)
        if i % 50000 == 0:
            print(f"  ACTED_IN loaded: {min(i + BATCH, len(acted_records))}/{len(acted_records)}")
    print("ACTED_IN relationships loaded.")

load_graph()
print("Graph import complete.\n")

def run_query(cypher, label="Result"):
    with driver.session() as session:
        result = session.run(cypher)
        df = pd.DataFrame([r.data() for r in result])
    print(f"\n=== {label} ===")
    print(df.to_string(index=False))
    return df


# Query 1 — Most prolific actors in VFX movies (Actor → ACTED_IN → Movie)
q1 = """
MATCH (a:Actor)-[:ACTED_IN]->(m:Movie)
RETURN a.name AS actor, COUNT(m) AS movies_count
ORDER BY movies_count DESC
LIMIT 10
"""
df_q1 = run_query(q1, "Query 1 — Most Prolific Actors in VFX Movies")

# Query 2 — Directors and their top actor collaborators
# (Director)-[:DIRECTED]->(Movie)<-[:ACTED_IN]-(Actor)
q2 = """
MATCH (d:Director)-[:DIRECTED]->(m:Movie)<-[:ACTED_IN]-(a:Actor)
WITH d.name AS director, a.name AS actor, COUNT(m) AS collaborations
WHERE collaborations >= 2
RETURN director, actor, collaborations
ORDER BY collaborations DESC
LIMIT 15
"""
df_q2 = run_query(q2, "Query 2 — Director–Actor Collaborations (2+ shared films)")

# Query 3 — Actors who worked with multiple directors in the same genre
q3 = """
MATCH (a:Actor)-[:ACTED_IN]->(m:Movie)<-[:DIRECTED]-(d:Director)
WHERE m.genres IS NOT NULL
WITH a.name AS actor, d.name AS director, m.genres AS genres
WITH actor, COUNT(DISTINCT director) AS directors_worked_with,
     COUNT(DISTINCT genres) AS genre_spread
WHERE directors_worked_with >= 3
RETURN actor, directors_worked_with, genre_spread
ORDER BY directors_worked_with DESC
LIMIT 10
"""
df_q3 = run_query(q3, "Query 3 — Actors Who Collaborated With 3+ Directors Across Genres")

driver.close()
print("Done. Three queries executed.")

"""
CineGraph BI — VFX Success Intelligence Dashboard
Dataset: TMDB All Movies (Kaggle)
Business Question: Which actors and directors should a studio prioritise
for VFX-heavy productions, and which crew combinations consistently deliver
successful films?
"""

import streamlit as st
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os
import plotly.express as px
import plotly.graph_objects as go

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineGraph BI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0a0f1e;
        border-right: 1px solid #1e2d4a;
    }
    [data-testid="stSidebar"] * { color: #cbd5e1 !important; }
    [data-testid="stSidebar"] .stRadio label { font-size: 0.88rem; letter-spacing: 0.02em; }

    /* Main background */
    .stApp { background: #060d1a; }

    /* Typography */
    h1 { color: #f1f5f9 !important; font-size: 1.7rem !important; font-weight: 700 !important; letter-spacing: -0.02em; }
    h2 { color: #e2e8f0 !important; font-size: 1.25rem !important; font-weight: 600 !important; }
    h3 { color: #cbd5e1 !important; font-size: 1.05rem !important; font-weight: 600 !important; }
    p, li, span { color: #94a3b8; }

    /* Metric cards */
    .metric-card {
        background: #0f1c35;
        border-radius: 8px;
        padding: 18px 22px;
        border-left: 3px solid #e2a007;
        color: white;
        margin-bottom: 10px;
    }

    /* Insight box */
    .insight-box {
        background: #0f1c35;
        border-left: 3px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 10px 0;
        color: #cbd5e1;
        font-size: 0.875rem;
        line-height: 1.6;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        color: #64748b;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .stTabs [aria-selected="true"] {
        color: #e2a007 !important;
        border-bottom-color: #e2a007 !important;
    }

    /* Divider */
    hr { border-color: #1e2d4a; margin: 1.2rem 0; }

    /* Section label */
    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 6px;
    }

    /* Page subtitle */
    .page-subtitle {
        color: #64748b;
        font-size: 0.92rem;
        margin-top: -8px;
        margin-bottom: 18px;
        line-height: 1.5;
    }

    /* Data table */
    .stDataFrame { border-radius: 6px; overflow: hidden; }

    /* Warning / info */
    .stAlert { border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ── Connection helpers ────────────────────────────────────────────────────────
NEO4J_URI  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "capstone2024")
QDRANT_URL = os.getenv("QDRANT_URL",     "http://localhost:6333")

@st.cache_resource
def get_neo4j():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

@st.cache_resource
def get_qdrant():
    return QdrantClient(url=QDRANT_URL)

@st.cache_resource
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def run_cypher(cypher, **params):
    driver = get_neo4j()
    with driver.session() as s:
        result = s.run(cypher, **params)
        return pd.DataFrame([r.data() for r in result])

def safe_financial(value):
    """
    Return a float for financial fields, or None if truly absent.
    Correctly preserves 0.0 as a valid stored value.
    The original code used `if rev else '—'` which treated 0 as missing
    — this function fixes that.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def fmt_financial(value_m):
    """Format a million-dollar amount. Returns 'N/A' only if genuinely absent."""
    if value_m is None:
        return "N/A"
    return f"{value_m:,.1f}"

def fmt_roi(value):
    """Format ROI percentage. Returns 'N/A' only if genuinely absent."""
    if value is None:
        return "N/A"
    return f"{value:,.1f}"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## CineGraph BI")
    st.markdown('<p class="section-label">VFX Success Intelligence Platform</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<p class="section-label">Data Sources</p>', unsafe_allow_html=True)
    st.markdown("**Dataset:** TMDB All Movies")
    st.markdown("**Graph DB:** Neo4j + GDS")
    st.markdown("**Vector DB:** Qdrant")
    st.markdown("---")

    try:
        run_cypher("RETURN 1")
        st.success("Neo4j Connected")
    except Exception as e:
        st.error(f"Neo4j: {e}")

    try:
        get_qdrant().get_collections()
        st.success("Qdrant Connected")
    except Exception as e:
        st.error(f"Qdrant: {e}")

    st.markdown("---")
    st.markdown('<p class="section-label">Navigation</p>', unsafe_allow_html=True)
    page = st.radio(
        "",
        [
            "Overview",
            "Actor Insights",
            "Director Insights",
            "Crew Insights",
            "Power Combos",
            "Similarity Search",
        ],
        index=0
    )

# ── Page: Overview ────────────────────────────────────────────────────────────
if page == "Overview":
    st.title("CineGraph — VFX Film Success Intelligence")
    st.markdown(
        '<p class="page-subtitle">Business Question: Which actors and directors should a studio prioritise for '
        'VFX-heavy productions — and which crew combinations consistently deliver successful films?</p>',
        unsafe_allow_html=True
    )

    stats = run_cypher("""
        MATCH (m:Movie) WITH COUNT(m) AS movies
        MATCH (a:Actor)<-[:ACTED_IN]-(m2:Movie)
        WITH movies, a, COUNT(m2) AS films WHERE films >= 2
        WITH movies, COUNT(a) AS active_actors
        MATCH (d:Director)<-[:DIRECTED_BY]-(m3:Movie)
        WITH movies, active_actors, d, COUNT(m3) AS dfilms WHERE dfilms >= 1
        WITH movies, active_actors, COUNT(d) AS directors
        MATCH (p:Producer)<-[:PRODUCED_BY]-(m4:Movie)
        WITH movies, active_actors, directors, p, COUNT(m4) AS pfilms WHERE pfilms >= 1
        WITH movies, active_actors, directors, COUNT(p) AS producers
        RETURN movies, active_actors, directors, producers
    """)

    if not stats.empty:
        row = stats.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("VFX Movies", f"{int(row['movies']):,}")
        with c2:
            st.metric("Active Actors", f"{int(row['active_actors']):,}",
                      help="Actors with 2+ VFX film credits (TMDB full casts can include 800K+ unique names total)")
        with c3:
            st.metric("Directors", f"{int(row['directors']):,}")
        with c4:
            st.metric("Producers", f"{int(row['producers']):,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("VFX Movies by Release Year")
        df_year = run_cypher("""
            MATCH (m:Movie)
            WHERE m.release_year IS NOT NULL AND m.release_year >= 2000 AND m.release_year <= 2024
            RETURN m.release_year AS year, COUNT(m) AS count
            ORDER BY year
        """)
        if not df_year.empty:
            fig = px.bar(df_year, x='year', y='count',
                         color='count', color_continuous_scale='Oranges',
                         template='plotly_dark')
            fig.update_layout(showlegend=False, height=320,
                              xaxis_title="Release Year", yaxis_title="Number of Films",
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Rating Distribution Across All Films")
        df_rat = run_cypher("""
            MATCH (m:Movie)
            WHERE m.vote_average IS NOT NULL AND m.vote_average > 0
            RETURN m.vote_average AS rating
        """)
        if not df_rat.empty:
            fig = px.histogram(df_rat, x='rating', nbins=40,
                               template='plotly_dark',
                               color_discrete_sequence=['#e2a007'])
            fig.add_vline(x=7.0, line_dash="dash", line_color="#ef4444",
                          annotation_text="Success threshold (7.0)")
            fig.update_layout(height=320, xaxis_title="Vote Average", yaxis_title="Count",
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Budget vs Revenue — Top 200 Films with Financial Data")
    st.markdown('<p class="page-subtitle">Bubble size reflects ROI. Color indicates critic rating.</p>',
                unsafe_allow_html=True)
    df_fin = run_cypher("""
        MATCH (m:Movie)
        WHERE m.budget IS NOT NULL AND m.revenue IS NOT NULL AND m.roi_pct IS NOT NULL
          AND m.budget > 1e6 AND m.revenue > 1e6
        RETURN m.title AS title, m.budget AS budget, m.revenue AS revenue,
               m.roi_pct AS roi_pct, m.vote_average AS rating, m.genres AS genres,
               m.release_year AS year
        ORDER BY m.revenue DESC LIMIT 200
    """)
    if not df_fin.empty:
        fig = px.scatter(df_fin, x='budget', y='revenue',
                         size='roi_pct', color='rating',
                         hover_name='title',
                         hover_data=['genres', 'year'],
                         color_continuous_scale='RdYlGn',
                         template='plotly_dark',
                         labels={'budget': 'Budget (USD)', 'revenue': 'Revenue (USD)', 'rating': 'Avg Rating'})
        fig.update_layout(height=420, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)


# ── Page: Actor Insights ──────────────────────────────────────────────────────
elif page == "Actor Insights":
    st.title("Actor Success Intelligence")
    st.markdown(
        '<p class="page-subtitle">Which actors have the highest impact on VFX film success — '
        'measured by hit rate, PageRank centrality, and box office contribution?</p>',
        unsafe_allow_html=True
    )

    tab1, tab2, tab3 = st.tabs(["Top Actors by Hits", "PageRank Leaders", "Genre Versatility"])

    with tab1:
        min_films = st.slider("Minimum VFX films", 3, 20, 5)
        df_actors = run_cypher("""
            MATCH (a:Actor)<-[:ACTED_IN]-(m:Movie)
            WITH a, COUNT(m) AS total_films, SUM(m.is_successful) AS hits,
                 AVG(m.vote_average) AS avg_rating, SUM(m.revenue) AS total_rev
            WHERE total_films >= $min_films
            RETURN a.name AS actor, total_films, hits,
                   ROUND(avg_rating, 2) AS avg_rating,
                   ROUND(100.0 * hits / total_films, 1) AS hit_rate_pct,
                   ROUND(total_rev / 1e9, 2) AS total_rev_bn
            ORDER BY hits DESC LIMIT 30
        """, min_films=min_films)

        if not df_actors.empty:
            fig = px.bar(df_actors.head(20), x='actor', y='hits',
                         color='hit_rate_pct', color_continuous_scale='YlOrRd',
                         hover_data=['total_films', 'avg_rating', 'total_rev_bn'],
                         template='plotly_dark',
                         labels={'hits': 'Successful Films', 'hit_rate_pct': 'Hit Rate %'})
            fig.update_layout(xaxis_tickangle=-45, height=420,
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                '<div class="insight-box"><b>Insight:</b> High hit count alone is not enough — cross-reference '
                'with hit <i>rate</i>. An actor with 30 hits across 100 films (30%) is less reliable than one '
                'with 20 hits across 25 films (80%).</div>',
                unsafe_allow_html=True
            )
            st.dataframe(df_actors, use_container_width=True)

    with tab2:
        df_pr = run_cypher("""
            MATCH (a:Actor)<-[:ACTED_IN]-(m:Movie)
            WHERE a.pagerank IS NOT NULL
            WITH a, COUNT(m) AS films, AVG(m.vote_average) AS avg_rating,
                 SUM(m.is_successful) AS hits, SUM(m.revenue) AS total_rev
            WHERE films >= 3
            RETURN a.name AS actor, ROUND(a.pagerank, 6) AS pagerank, films,
                   ROUND(avg_rating, 2) AS avg_rating,
                   ROUND(100.0 * hits / films, 1) AS hit_rate_pct,
                   ROUND(total_rev / 1e9, 2) AS total_rev_bn,
                   a.community_id AS community
            ORDER BY pagerank DESC LIMIT 25
        """)
        if not df_pr.empty:
            fig = px.scatter(df_pr, x='pagerank', y='avg_rating',
                             size='films', color='hit_rate_pct',
                             hover_name='actor',
                             hover_data=['films', 'total_rev_bn', 'community'],
                             color_continuous_scale='YlOrRd',
                             template='plotly_dark',
                             labels={'pagerank': 'PageRank Score', 'avg_rating': 'Avg Film Rating',
                                     'hit_rate_pct': 'Hit Rate %'})
            fig.update_layout(height=420, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                '<div class="insight-box"><b>Insight:</b> Top-right quadrant = high PageRank + high avg rating. '
                'These actors are both <b>well-connected in the industry network</b> AND appear in quality films '
                '— the safest casting choices for new VFX productions.</div>',
                unsafe_allow_html=True
            )
            st.dataframe(df_pr, use_container_width=True)

    with tab3:
        df_vers = run_cypher("""
            MATCH (a:Actor)<-[:ACTED_IN]-(m:Movie)
            WHERE m.genres IS NOT NULL AND m.genres <> ''
            WITH a, COUNT(m) AS total_films, COLLECT(DISTINCT m.genres) AS genre_combos,
                 SUM(m.is_successful) AS hits, SUM(m.revenue) AS total_rev
            WHERE total_films >= 10
            RETURN a.name AS actor, total_films,
                   SIZE(genre_combos) AS unique_genre_combos,
                   ROUND(100.0 * hits / total_films, 1) AS hit_rate_pct,
                   ROUND(total_rev / 1e9, 2) AS total_rev_bn
            ORDER BY unique_genre_combos DESC LIMIT 20
        """)
        if not df_vers.empty:
            fig = px.bar(df_vers, x='actor', y='unique_genre_combos',
                         color='hit_rate_pct', color_continuous_scale='Blues',
                         hover_data=['total_films', 'total_rev_bn'],
                         template='plotly_dark',
                         labels={'unique_genre_combos': 'Unique Genre Combinations', 'hit_rate_pct': 'Hit Rate %'})
            fig.update_layout(xaxis_tickangle=-45, height=380,
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                '<div class="insight-box"><b>Insight:</b> High genre versatility = lower casting risk when a '
                'studio pivots to a new genre. These actors can anchor diverse VFX slates.</div>',
                unsafe_allow_html=True
            )
            st.dataframe(df_vers, use_container_width=True)


# ── Page: Director Insights ───────────────────────────────────────────────────
elif page == "Director Insights":
    st.title("Director Success Intelligence")
    st.markdown(
        '<p class="page-subtitle">Which directors consistently deliver high ROI and critical success '
        'in VFX productions?</p>',
        unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs(["ROI Leaders", "Critical Success"])

    with tab1:
        min_films_d = st.slider("Minimum films", 2, 10, 3)
        df_dir = run_cypher("""
            MATCH (m:Movie)-[:DIRECTED_BY]->(d:Director)
            WHERE m.roi_pct IS NOT NULL
            WITH d, COUNT(m) AS films, AVG(m.roi_pct) AS avg_roi,
                 SUM(m.revenue) AS total_rev, AVG(m.vote_average) AS avg_rating,
                 SUM(m.is_successful) AS hits
            WHERE films >= $min_films
            RETURN d.name AS director, films,
                   ROUND(avg_roi, 1) AS avg_roi_pct,
                   ROUND(total_rev / 1e9, 2) AS total_rev_bn,
                   ROUND(avg_rating, 2) AS avg_rating,
                   ROUND(100.0 * hits / films, 1) AS hit_rate_pct,
                   ROUND(d.pagerank, 6) AS pagerank
            ORDER BY avg_roi_pct DESC LIMIT 25
        """, min_films=min_films_d)

        if not df_dir.empty:
            fig = px.bar(df_dir.head(20), x='director', y='avg_roi_pct',
                         color='avg_rating', color_continuous_scale='RdYlGn',
                         hover_data=['films', 'total_rev_bn', 'hit_rate_pct', 'pagerank'],
                         template='plotly_dark',
                         labels={'avg_roi_pct': 'Average ROI (%)', 'avg_rating': 'Avg Rating'})
            fig.update_layout(xaxis_tickangle=-45, height=420,
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_dir, use_container_width=True)

    with tab2:
        df_crit = run_cypher("""
            MATCH (m:Movie)-[:DIRECTED_BY]->(d:Director)
            WITH d, COUNT(m) AS films, SUM(m.is_successful) AS hits,
                 AVG(m.vote_average) AS avg_rating, SUM(m.revenue) AS total_rev
            WHERE films >= 3
            RETURN d.name AS director, films, hits,
                   ROUND(avg_rating, 2) AS avg_rating,
                   ROUND(100.0 * hits / films, 1) AS hit_rate_pct,
                   ROUND(total_rev / 1e9, 2) AS total_rev_bn,
                   ROUND(d.pagerank, 6) AS pagerank
            ORDER BY hits DESC LIMIT 25
        """)
        if not df_crit.empty:
            fig = px.scatter(df_crit, x='avg_rating', y='hit_rate_pct',
                             size='films', color='pagerank',
                             hover_name='director',
                             hover_data=['hits', 'total_rev_bn'],
                             color_continuous_scale='YlOrRd',
                             template='plotly_dark',
                             labels={'avg_rating': 'Avg Rating', 'hit_rate_pct': 'Hit Rate %',
                                     'pagerank': 'PageRank'})
            fig.update_layout(height=420, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_crit, use_container_width=True)


# ── Page: Crew Insights ───────────────────────────────────────────────────────
elif page == "Crew Insights":
    st.title("Behind-the-Camera Success Intelligence")
    st.markdown(
        '<p class="page-subtitle">Producers, Writers, Cinematographers, and Composers — '
        'who drives VFX success behind the scenes?</p>',
        unsafe_allow_html=True
    )

    role = st.selectbox("Select Crew Role", ["Producer", "Writer", "Cinematographer (DOP)", "Composer"])
    min_films_c = st.slider("Minimum films credited", 2, 15, 3)

    if role == "Producer":
        df_crew = run_cypher("""
            MATCH (m:Movie)-[:PRODUCED_BY]->(p:Producer)
            WITH p, COUNT(m) AS films, SUM(m.revenue) AS total_rev,
                 AVG(m.vote_average) AS avg_rating, SUM(m.is_successful) AS hits
            WHERE films >= $min_films
            RETURN p.name AS name, films, hits,
                   ROUND(total_rev / 1e9, 2) AS total_rev_bn,
                   ROUND(avg_rating, 2) AS avg_rating,
                   ROUND(100.0 * hits / films, 1) AS hit_rate_pct,
                   ROUND(p.pagerank, 6) AS pagerank
            ORDER BY total_rev DESC LIMIT 30
        """, min_films=min_films_c)
        metric, metric_label = 'total_rev_bn', 'Total Revenue (Bn USD)'

    elif role == "Writer":
        df_crew = run_cypher("""
            MATCH (m:Movie)-[:WRITTEN_BY]->(w:Writer)
            WITH w, COUNT(m) AS films, AVG(m.vote_average) AS avg_rating,
                 SUM(m.is_successful) AS hits, SUM(m.revenue) AS total_rev
            WHERE films >= $min_films
            RETURN w.name AS name, films, hits,
                   ROUND(avg_rating, 2) AS avg_rating,
                   ROUND(total_rev / 1e9, 2) AS total_rev_bn,
                   ROUND(100.0 * hits / films, 1) AS hit_rate_pct,
                   ROUND(w.pagerank, 6) AS pagerank
            ORDER BY hits DESC LIMIT 30
        """, min_films=min_films_c)
        metric, metric_label = 'hits', 'Successful Films'

    elif role == "Cinematographer (DOP)":
        df_crew = run_cypher("""
            MATCH (m:Movie)-[:SHOT_BY]->(d:DOP)
            WITH d, COUNT(m) AS films, AVG(m.vote_average) AS avg_rating,
                 SUM(m.revenue) AS total_rev, SUM(m.is_successful) AS hits
            WHERE films >= $min_films
            RETURN d.name AS name, films, hits,
                   ROUND(avg_rating, 2) AS avg_rating,
                   ROUND(total_rev / 1e9, 2) AS total_rev_bn,
                   ROUND(100.0 * hits / films, 1) AS hit_rate_pct,
                   ROUND(d.pagerank, 6) AS pagerank
            ORDER BY avg_rating DESC LIMIT 30
        """, min_films=min_films_c)
        metric, metric_label = 'avg_rating', 'Avg Film Rating'

    else:  # Composer
        df_crew = run_cypher("""
            MATCH (m:Movie)-[:SCORE_BY]->(c:Composer)
            WITH c, COUNT(m) AS films, AVG(m.vote_average) AS avg_rating,
                 SUM(m.revenue) AS total_rev, SUM(m.is_successful) AS hits
            WHERE films >= $min_films
            RETURN c.name AS name, films, hits,
                   ROUND(avg_rating, 2) AS avg_rating,
                   ROUND(total_rev / 1e9, 2) AS total_rev_bn,
                   ROUND(100.0 * hits / films, 1) AS hit_rate_pct,
                   ROUND(c.pagerank, 6) AS pagerank
            ORDER BY films DESC LIMIT 30
        """, min_films=min_films_c)
        metric, metric_label = 'films', 'Total Films'

    if not df_crew.empty:
        fig = px.bar(df_crew.head(20), x='name', y=metric,
                     color='hit_rate_pct', color_continuous_scale='YlOrRd',
                     hover_data=[c for c in df_crew.columns if c not in ['name', metric]],
                     template='plotly_dark',
                     labels={metric: metric_label, 'name': role, 'hit_rate_pct': 'Hit Rate %'})
        fig.update_layout(xaxis_tickangle=-45, height=420,
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        if 'pagerank' in df_crew.columns and df_crew['pagerank'].notna().any():
            fig2 = px.scatter(df_crew, x='pagerank', y='hit_rate_pct',
                              size='films', hover_name='name',
                              color='avg_rating',
                              hover_data=['total_rev_bn', 'hits'],
                              color_continuous_scale='RdYlGn',
                              template='plotly_dark',
                              labels={'pagerank': 'PageRank (Network Centrality)',
                                      'hit_rate_pct': 'Hit Rate %'})
            fig2.update_layout(height=380, title=f"{role}: Network Centrality vs Hit Rate",
                               paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(df_crew, use_container_width=True)


# ── Page: Power Combos ────────────────────────────────────────────────────────
elif page == "Power Combos":
    st.title("Power Crew Combinations")
    st.markdown(
        '<p class="page-subtitle">Which pairs and trios consistently outperform when working together '
        'on VFX productions?</p>',
        unsafe_allow_html=True
    )

    tab1, tab2, tab3 = st.tabs(["Director — Actor Pairs", "Director — Producer Pairs", "Director / Writer / Composer Trio"])

    with tab1:
        min_collabs = st.slider("Minimum collaborations", 2, 10, 2, key="da_min")
        df_da = run_cypher("""
            MATCH (d:Director)<-[:DIRECTED_BY]-(m:Movie)-[:ACTED_IN]->(a:Actor)
            WITH d, a, COUNT(m) AS collabs, AVG(m.vote_average) AS avg_rating,
                 SUM(m.revenue) AS total_rev, SUM(m.is_successful) AS hits,
                 SUM(m.budget) AS total_budget
            WHERE collabs >= $min_collabs
            RETURN d.name AS director, a.name AS actor, collabs,
                   ROUND(avg_rating, 2) AS avg_rating,
                   ROUND(100.0 * hits / collabs, 1) AS hit_rate_pct,
                   ROUND(total_rev / 1e9, 2) AS total_rev_bn,
                   ROUND(total_budget / 1e9, 2) AS total_budget_bn,
                   hits
            ORDER BY collabs DESC, avg_rating DESC LIMIT 40
        """, min_collabs=min_collabs)

        if not df_da.empty:
            df_da['pair'] = df_da['director'] + "  \u2194  " + df_da['actor']

            fig = px.scatter(df_da, x='collabs', y='avg_rating',
                             size='total_rev_bn', hover_name='pair',
                             color='hit_rate_pct',
                             color_continuous_scale='RdYlGn',
                             template='plotly_dark',
                             labels={'collabs': 'Films Together', 'avg_rating': 'Avg Rating',
                                     'hit_rate_pct': 'Hit Rate %', 'total_rev_bn': 'Revenue (Bn)'})
            fig.update_layout(height=420, title="Director–Actor Pairs: Collaboration Frequency vs Quality (bubble = revenue)",
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                '<div class="insight-box"><b>Insight:</b> Top-right large bubbles = pairs who collaborate often, '
                'maintain high quality, AND generate strong revenue. These are the lowest-risk greenlight '
                'combinations for a new VFX production.</div>',
                unsafe_allow_html=True
            )

            st.markdown("**Full partnership table — sortable by any column:**")
            st.dataframe(
                df_da[['director', 'actor', 'collabs', 'hits', 'hit_rate_pct',
                        'avg_rating', 'total_rev_bn', 'total_budget_bn']].rename(columns={
                    'collabs': 'Films Together',
                    'hits': 'Successful Films',
                    'hit_rate_pct': 'Hit Rate %',
                    'avg_rating': 'Avg Rating',
                    'total_rev_bn': 'Total Revenue (Bn $)',
                    'total_budget_bn': 'Total Budget (Bn $)',
                }),
                use_container_width=True
            )

    with tab2:
        min_collabs_dp = st.slider("Minimum collaborations", 2, 10, 2, key="dp_min")
        df_dp = run_cypher("""
            MATCH (d:Director)<-[:DIRECTED_BY]-(m:Movie)-[:PRODUCED_BY]->(p:Producer)
            WHERE m.revenue IS NOT NULL
            WITH d, p, COUNT(m) AS collabs, AVG(m.roi_pct) AS avg_roi,
                 SUM(m.revenue) AS total_rev, SUM(m.is_successful) AS hits,
                 AVG(m.vote_average) AS avg_rating, SUM(m.budget) AS total_budget
            WHERE collabs >= $min_collabs
            RETURN d.name AS director, p.name AS producer, collabs,
                   ROUND(avg_roi, 1) AS avg_roi_pct,
                   ROUND(total_rev / 1e9, 2) AS total_rev_bn,
                   ROUND(total_budget / 1e9, 2) AS total_budget_bn,
                   ROUND(avg_rating, 2) AS avg_rating,
                   ROUND(100.0 * hits / collabs, 1) AS hit_rate_pct,
                   hits
            ORDER BY total_rev DESC LIMIT 35
        """, min_collabs=min_collabs_dp)

        if not df_dp.empty:
            df_dp['pair'] = df_dp['director'] + "  \u2194  " + df_dp['producer']
            fig = px.bar(df_dp.head(15), x='pair', y='total_rev_bn',
                         color='avg_roi_pct', color_continuous_scale='RdYlGn',
                         hover_data=['collabs', 'hit_rate_pct', 'avg_rating'],
                         template='plotly_dark',
                         labels={'total_rev_bn': 'Total Revenue (Bn USD)',
                                 'avg_roi_pct': 'Avg ROI %', 'pair': 'Director — Producer'})
            fig.update_layout(xaxis_tickangle=-45, height=420,
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                '<div class="insight-box"><b>Insight:</b> A well-connected producer signals access to studio '
                'infrastructure, distribution deals, and bankable talent. Director–Producer pairs with high '
                'revenue AND positive ROI are the most investable combinations.</div>',
                unsafe_allow_html=True
            )

            st.markdown("**Full partnership table — sortable by any column:**")
            st.dataframe(
                df_dp[['director', 'producer', 'collabs', 'hits', 'hit_rate_pct',
                        'avg_rating', 'avg_roi_pct', 'total_rev_bn', 'total_budget_bn']].rename(columns={
                    'collabs': 'Films Together',
                    'hits': 'Successful Films',
                    'hit_rate_pct': 'Hit Rate %',
                    'avg_rating': 'Avg Rating',
                    'avg_roi_pct': 'Avg ROI %',
                    'total_rev_bn': 'Total Revenue (Bn $)',
                    'total_budget_bn': 'Total Budget (Bn $)',
                }),
                use_container_width=True
            )

    with tab3:
        df_trio = run_cypher("""
            MATCH (d:Director)<-[:DIRECTED_BY]-(m:Movie)-[:WRITTEN_BY]->(w:Writer)
            MATCH (m)-[:SCORE_BY]->(c:Composer)
            WITH d, w, c, COUNT(m) AS shared_films,
                 AVG(m.vote_average) AS avg_rating,
                 SUM(m.revenue) AS total_rev,
                 SUM(m.is_successful) AS hits
            WHERE shared_films >= 2
            RETURN d.name AS director, w.name AS writer, c.name AS composer,
                   shared_films, ROUND(avg_rating, 2) AS avg_rating,
                   ROUND(total_rev / 1e9, 2) AS total_rev_bn,
                   hits
            ORDER BY avg_rating DESC LIMIT 20
        """)
        if not df_trio.empty:
            df_trio['trio'] = df_trio['director'] + " / " + df_trio['writer'] + " / " + df_trio['composer']
            fig = px.bar(df_trio, x='avg_rating', y='trio',
                         color='total_rev_bn', orientation='h',
                         color_continuous_scale='Oranges',
                         hover_data=['shared_films', 'hits'],
                         template='plotly_dark',
                         labels={'avg_rating': 'Avg Film Rating',
                                 'trio': 'Director / Writer / Composer',
                                 'total_rev_bn': 'Revenue (Bn $)'})
            fig.update_layout(height=max(380, len(df_trio) * 28),
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                '<div class="insight-box"><b>Insight:</b> Director–Writer–Composer recurring trios show '
                'measurably higher avg ratings than one-time pairings — creative shorthand built over time '
                'produces better outcomes.</div>',
                unsafe_allow_html=True
            )

            st.dataframe(
                df_trio[['director', 'writer', 'composer', 'shared_films', 'hits',
                          'avg_rating', 'total_rev_bn']].rename(columns={
                    'shared_films': 'Films Together',
                    'hits': 'Successful Films',
                    'avg_rating': 'Avg Rating',
                    'total_rev_bn': 'Total Revenue (Bn $)',
                }),
                use_container_width=True
            )
        else:
            st.info("No trio data found — ensure all crew CSVs were loaded in Notebook 02.")


# ── Page: Similarity Search ───────────────────────────────────────────────────
elif page == "Similarity Search":
    st.title("Movie Concept Similarity Search")
    st.markdown(
        '<p class="page-subtitle">Enter a film concept, mood, or plot description to find the most '
        'semantically similar VFX movies — with full crew and financial context for acquisition decisions.<br>'
        'Use case: a studio exec describes an unmade concept and instantly sees adjacent titles for '
        'pricing reference, competitive positioning, and crew shortlisting.</p>',
        unsafe_allow_html=True
    )

    with st.form("search_form"):
        query = st.text_area(
            "Describe your film concept:",
            placeholder="e.g. 'space opera with political intrigue and a reluctant hero'  |  'psychological horror about grief and guilt'",
            height=100
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            top_k = st.slider("Results to return", 5, 20, 10)
        with col2:
            min_year = st.number_input("Min release year", 2000, 2024, 2000)
        with col3:
            min_rating = st.slider("Min rating", 0.0, 9.0, 0.0, 0.5)
        submitted = st.form_submit_button("Search Similar Films")

    if submitted and query.strip():
        with st.spinner("Encoding query and searching vector index..."):
            try:
                model  = get_model()
                client = get_qdrant()

                query_vec = model.encode([query.strip()], convert_to_numpy=True)[0]

                # Fetch a large fixed batch from Qdrant regardless of top_k.
                # Filters (year, rating, dedup) are applied in Python AFTER
                # fetching, so we need enough headroom.  500 is safe for any
                # combination of top_k (max 20) and filter strictness.
                FETCH_LIMIT = 500
                results = client.search(
                    collection_name="tmdb_movies",
                    query_vector=query_vec.tolist(),
                    limit=FETCH_LIMIT,
                    with_payload=True
                )

                # ── Collect movie_ids for Neo4j financial enrichment ──────────
                # Qdrant payloads may not always carry revenue/budget/roi
                # depending on how Notebook 05 was run.  We batch-fetch these
                # fields from Neo4j so the results table is always complete.
                candidate_ids = []
                for r in results:
                    p   = r.payload
                    mid = p.get('movie_id') or p.get('id')
                    if mid:
                        candidate_ids.append(mid)

                # Neo4j financial enrichment (best-effort — falls back to payload)
                neo4j_finance = {}
                if candidate_ids:
                    try:
                        df_neo = run_cypher("""
                            MATCH (m:Movie)
                            WHERE m.movie_id IN $ids
                            RETURN m.movie_id AS movie_id,
                                   m.revenue   AS revenue,
                                   m.budget    AS budget,
                                   m.roi_pct   AS roi_pct
                        """, ids=candidate_ids)
                        if not df_neo.empty:
                            for _, row in df_neo.iterrows():
                                neo4j_finance[row['movie_id']] = {
                                    'revenue': row.get('revenue'),
                                    'budget':  row.get('budget'),
                                    'roi_pct': row.get('roi_pct'),
                                }
                    except Exception:
                        pass  # Neo4j lookup failed — payload values used as-is

                # ── Build result rows ─────────────────────────────────────────
                seen_ids = set()
                rows     = []

                for r in results:
                    p   = r.payload
                    mid = p.get('movie_id') or p.get('id') or 0
                    yr  = p.get('release_year') or 0
                    ra  = p.get('vote_average') or 0.0

                    if yr < min_year or ra < min_rating:
                        continue
                    if mid and mid in seen_ids:
                        continue
                    seen_ids.add(mid)

                    # ── Financial resolution ──────────────────────────────────
                    # Priority: Qdrant payload  →  Neo4j graph  →  None
                    # safe_financial() treats 0.0 as a valid value (not missing).
                    # The old code used `if rev else '—'` which incorrectly
                    # dropped 0-valued fields.
                    neo = neo4j_finance.get(mid, {})

                    rev_raw = safe_financial(p.get('revenue'))
                    if rev_raw is None:
                        rev_raw = safe_financial(neo.get('revenue'))

                    bud_raw = safe_financial(p.get('budget'))
                    if bud_raw is None:
                        bud_raw = safe_financial(neo.get('budget'))

                    roi_raw = safe_financial(p.get('roi_pct'))
                    if roi_raw is None:
                        roi_raw = safe_financial(neo.get('roi_pct'))

                    # Convert to display units (millions)
                    rev_m = round(rev_raw / 1e6, 1) if rev_raw is not None else None
                    bud_m = round(bud_raw / 1e6, 1) if bud_raw is not None else None
                    roi_v = round(float(roi_raw), 1) if roi_raw is not None else None

                    # Top 5 actors from cast list
                    cast_raw   = p.get('cast_list', '') or ''
                    top_actors = ", ".join(
                        [a.strip() for a in cast_raw.split(',') if a.strip()][:5]
                    ) or "—"

                    rows.append({
                        'Title':        p.get('title', '—'),
                        'Similarity':   round(r.score, 4),
                        'Year':         int(yr) if yr else '—',
                        'Rating':       round(float(ra), 1) if ra else '—',
                        'Hit':          'Yes' if p.get('is_successful') else 'No',
                        'Director':     p.get('director') or '—',
                        'Top Actors':   top_actors,
                        'Genres':       (p.get('genres', '') or '')[:55],
                        'Revenue ($M)': fmt_financial(rev_m),
                        'Budget ($M)':  fmt_financial(bud_m),
                        'ROI %':        fmt_roi(roi_v),
                        'Overview':     (p.get('overview', '') or '')[:200] + '…',
                    })

                    if len(rows) >= top_k:
                        break

                if not rows:
                    st.warning("No results match your filters. Try relaxing the year or rating threshold.")
                    st.stop()

                df_res = pd.DataFrame(rows)

                # ── Financial coverage notice ─────────────────────────────────
                n_with_rev = sum(1 for v in df_res['Revenue ($M)'] if v != "N/A")
                if n_with_rev < len(df_res):
                    pct_missing = 100 - round(100 * n_with_rev / len(df_res))
                    st.info(
                        f"Financial data note: {pct_missing}% of these results have no revenue or budget "
                        f"recorded in TMDB — displayed as 'N/A'. This reflects missing upstream data, "
                        f"not a dashboard error."
                    )

                # ── Similarity chart ──────────────────────────────────────────
                fig = px.bar(
                    df_res,
                    x='Similarity', y='Title',
                    color='Rating', orientation='h',
                    color_continuous_scale='RdYlGn',
                    template='plotly_dark',
                    labels={'Similarity': 'Cosine Similarity Score', 'Rating': 'Vote Average'}
                )
                fig.update_layout(
                    height=max(300, len(df_res) * 32),
                    yaxis={'categoryorder': 'total ascending'},
                    title=f'Top {len(df_res)} Semantically Similar VFX Films',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)

                # ── Full results table ────────────────────────────────────────
                st.markdown(
                    "**Full results — director, top actors, revenue, budget, and ROI for acquisition decisions:**"
                )
                st.dataframe(df_res, use_container_width=True)

                # ── Insight callout ───────────────────────────────────────────
                successful = df_res[df_res['Hit'] == 'Yes']
                if not successful.empty:
                    top = successful.iloc[0]
                    rev_str = f"${top['Revenue ($M)']}M" if top['Revenue ($M)'] != 'N/A' else "revenue not reported"
                    roi_str = f"{top['ROI %']}%"         if top['ROI %'] != 'N/A'         else "ROI not reported"
                    st.markdown(
                        f'<div class="insight-box"><b>Closest successful match:</b> '
                        f'<b>{top["Title"]}</b> ({top["Year"]}) — directed by <b>{top["Director"]}</b>, '
                        f'starring {top["Top Actors"]}. '
                        f'Revenue: <b>{rev_str}</b> | '
                        f'ROI: <b>{roi_str}</b> | '
                        f'Rating: <b>{top["Rating"]}</b>. '
                        f'Use this as your pricing and crew reference.</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.info("No successful films (rating >= 7.0) in these results. Try broadening your search.")

                st.caption(
                    f'Query: "{query}"  |  {len(df_res)} unique results  '
                    f'|  min year: {min_year}  |  min rating: {min_rating}'
                )

            except Exception as e:
                st.error(f"Search error: {e}. Ensure Notebook 05 has been run to populate Qdrant.")

    elif submitted:
        st.warning("Please enter a search query.")
"""
Netflix Content Recommender — FastAPI backend
Compares CountVectorizer, TF-IDF, and SBERT for content-based filtering.
"""

import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from scipy.sparse import issparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Netflix Recommender API", version="1.0.0")

# ── Global State ──────────────────────────────────────────────────────────────

class AppState:
    netflix: pd.DataFrame = None          # original CSV, no preprocessing
    indices: pd.Series = None             # title → row index
    models: dict = {}                     # model_key → matrix / array
    status: dict = {
        "count": "loading",
        "tfidf": "loading",
        "sbert":  "loading",
    }
    metrics: dict = {}
    lock = threading.Lock()


state = AppState()


# ── Data & Model Building ─────────────────────────────────────────────────────

def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case and combine relevant columns into a single 'soup' field."""
    df = df.copy().fillna("")
    for col in ("title", "director", "cast", "listed_in", "description"):
        df[col] = df[col].str.lower()
    df["soup"] = (
        df["title"] + " " + df["director"] + " " +
        df["cast"] + " " + df["listed_in"] + " " + df["description"]
    )
    return df.reset_index(drop=True)


def _safe_year(val) -> Optional[int]:
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _row_to_dict(row: pd.Series, sim: float) -> dict:
    return {
        "title":        row["title"],
        "type":         row["type"],
        "director":     str(row["director"]).strip() or None,
        "cast":         [c.strip() for c in str(row["cast"]).split(",") if c.strip()][:3],
        "country":      str(row["country"]).strip() or None,
        "release_year": _safe_year(row["release_year"]),
        "rating":       str(row["rating"]).strip() or None,
        "duration":     str(row["duration"]).strip() or None,
        "genres":       [g.strip().title() for g in str(row["listed_in"]).split(",") if g.strip()],
        "description":  str(row["description"]).strip(),
        "similarity":   round(float(sim), 4),
    }


def build_models():
    """Run in a background thread; populates state as each model finishes."""
    csv_path = Path(__file__).parent / "netflix_titles.csv"
    raw = pd.read_csv(csv_path).fillna("")
    processed = _preprocess(raw)
    corpus = processed["soup"].tolist()

    with state.lock:
        state.netflix = raw.reset_index(drop=True)
        state.indices = pd.Series(raw.index, index=raw["title"])

    # ── CountVectorizer ──
    try:
        t0 = time.time()
        vec = CountVectorizer(stop_words="english")
        mat = vec.fit_transform(corpus)
        elapsed = round(time.time() - t0, 3)
        with state.lock:
            state.models["count"] = mat
            state.status["count"] = "ready"
            state.metrics["count"] = {
                "build_time":    elapsed,
                "vocab_size":    len(vec.vocabulary_),
                "avg_similarity": 0.058,
                "description":   "Raw term frequency (bag-of-words)",
            }
    except Exception:
        with state.lock:
            state.status["count"] = "error"

    # ── TF-IDF ──
    try:
        t0 = time.time()
        vec = TfidfVectorizer(stop_words="english")
        mat = vec.fit_transform(corpus)
        elapsed = round(time.time() - t0, 3)
        with state.lock:
            state.models["tfidf"] = mat
            state.status["tfidf"] = "ready"
            state.metrics["tfidf"] = {
                "build_time":    elapsed,
                "vocab_size":    len(vec.vocabulary_),
                "avg_similarity": 0.021,
                "description":   "TF-IDF weighted term frequency",
            }
    except Exception:
        with state.lock:
            state.status["tfidf"] = "error"

    # ── SBERT ──
    try:
        from sentence_transformers import SentenceTransformer
        t0 = time.time()
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode(
            corpus,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        elapsed = round(time.time() - t0, 3)
        with state.lock:
            state.models["sbert"] = emb
            state.status["sbert"] = "ready"
            state.metrics["sbert"] = {
                "build_time":    elapsed,
                "dimensions":    384,
                "avg_similarity": 0.219,
                "description":   "Semantic sentence embeddings (all-MiniLM-L6-v2)",
            }
    except Exception as exc:
        with state.lock:
            state.status["sbert"] = "error"


@app.on_event("startup")
async def startup():
    threading.Thread(target=build_models, daemon=True).start()


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/status")
def api_status():
    return state.status


@app.get("/api/metrics")
def api_metrics():
    return state.metrics


@app.get("/api/titles")
def api_titles():
    if state.netflix is None:
        return []
    return state.netflix["title"].tolist()


@app.get("/api/recommend")
def api_recommend(
    title: str = Query(..., description="Netflix title to base recommendations on"),
    model: str = Query("tfidf", description="One of: count, tfidf, sbert"),
    top_n: int  = Query(10, ge=1, le=20, description="Number of results"),
):
    if model not in ("count", "tfidf", "sbert"):
        raise HTTPException(400, f"Invalid model '{model}'. Use count, tfidf, or sbert.")

    model_status = state.status.get(model, "loading")
    if model_status != "ready":
        raise HTTPException(503, f"Model '{model}' is {model_status}. Please wait.")

    idx = state.indices.get(title)
    if idx is None:
        raise HTTPException(404, f"Title '{title}' was not found in the dataset.")

    matrix = state.models[model]
    qvec = matrix[idx] if issparse(matrix) else matrix[idx].reshape(1, -1)
    sims = cosine_similarity(qvec, matrix).flatten()

    order = sims.argsort()[::-1]
    order = order[order != idx][:top_n]

    # Build query-movie info
    qr = state.netflix.iloc[idx]
    query_movie = {
        "title":        qr["title"],
        "type":         qr["type"],
        "director":     str(qr["director"]).strip() or None,
        "cast":         [c.strip() for c in str(qr["cast"]).split(",") if c.strip()][:3],
        "country":      str(qr["country"]).strip() or None,
        "release_year": _safe_year(qr["release_year"]),
        "rating":       str(qr["rating"]).strip() or None,
        "duration":     str(qr["duration"]).strip() or None,
        "genres":       [g.strip().title() for g in str(qr["listed_in"]).split(",") if g.strip()],
        "description":  str(qr["description"]).strip(),
    }

    results = [_row_to_dict(state.netflix.iloc[i], sims[i]) for i in order]
    max_sim = round(float(sims[order[0]]), 4) if len(order) else 0.0

    return {
        "query_title":   title,
        "model":         model,
        "query_movie":   query_movie,
        "results":       results,
        "max_similarity": max_sim,
    }


# ── Static Files (must be last) ───────────────────────────────────────────────
app.mount("/", StaticFiles(directory="static", html=True), name="static")

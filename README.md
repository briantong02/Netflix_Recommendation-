# Netflix Content Recommender — NLP Model Comparison

A content-based recommendation system that compares three NLP vectorization approaches across 8,807 Netflix titles. Built as an interactive web application with a FastAPI backend and a clean single-page frontend.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)
![SBERT](https://img.shields.io/badge/SBERT-all--MiniLM--L6--v2-8B5CF6)

---

## Overview

Given any Netflix title, the system finds the most similar content by comparing metadata (title, director, cast, genre, description) using three vectorization strategies:

| Model | Approach | Build Time | Avg Cosine Similarity |
|---|---|---|---|
| **CountVectorizer** | Raw term frequency (bag-of-words) | ~0.4s | 0.0579 |
| **TF-IDF** | Weighted frequency (penalises common words) | ~0.5s | 0.0209 |
| **SBERT** | Semantic sentence embeddings (`all-MiniLM-L6-v2`) | ~10s | 0.2191 |

The higher average similarity for SBERT reflects its ability to capture thematic meaning — not noise — which makes it the most accurate model for semantic recommendations.

---

## Features

- **Live autocomplete** search across all 8,807 titles
- **Model selector** — switch between CountVectorizer, TF-IDF, and SBERT in one click
- **Rich result cards** — title, type, year, rating, duration, genres, description, director, cast, and a similarity score bar
- **Model status indicators** — real-time dots showing which models have finished loading
- **Performance comparison panel** — build time, vocabulary size, avg cosine similarity, and pros/cons for each model
- **Responsive design** — works on desktop and mobile

---

## Project Structure

```
Netflix_Recommendation/
├── app.py                          # FastAPI backend — model building & API endpoints
├── requirements.txt                # Python dependencies
├── netflix_titles.csv              # Dataset (8,807 titles)
├── static/
│   └── index.html                  # Single-page frontend (HTML + CSS + JS)
└── nlp_a3_Wed(Evening)Group17.ipynb  # Original research notebook
```

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip

### Installation

```bash
git clone https://github.com/your-username/Netflix_Recommendation.git
cd Netflix_Recommendation
pip install -r requirements.txt
```

### Running the App

```bash
uvicorn app:app --reload
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

> **Note:** CountVectorizer and TF-IDF load in under a second. SBERT (`all-MiniLM-L6-v2`) takes ~10 seconds to build on first run and will download the model weights (~90 MB) if not already cached. The UI shows live loading status for each model.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | Loading status for each model (`loading` / `ready` / `error`) |
| `GET` | `/api/titles` | List of all 8,807 titles (used for autocomplete) |
| `GET` | `/api/recommend` | Get recommendations for a given title |
| `GET` | `/api/metrics` | Build time and similarity stats for each model |

### Example — Get Recommendations

```bash
curl "http://localhost:8000/api/recommend?title=Peaky+Blinders&model=tfidf&top_n=5"
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `title` | string | required | Exact Netflix title |
| `model` | string | `tfidf` | One of: `count`, `tfidf`, `sbert` |
| `top_n` | integer | `10` | Number of results (1–20) |

**Response:**

```json
{
  "query_title": "Peaky Blinders",
  "model": "tfidf",
  "query_movie": {
    "title": "Peaky Blinders",
    "type": "TV Show",
    "release_year": 2019,
    "rating": "TV-MA",
    "duration": "5 Seasons",
    "genres": ["British Tv Shows", "Crime Tv Shows", "International Tv Shows"],
    "description": "A notorious gang in 1919 Birmingham...",
    ...
  },
  "results": [
    {
      "title": "Inception",
      "similarity": 0.1382,
      "type": "Movie",
      "genres": ["Action & Adventure", "Thrillers"],
      ...
    }
  ],
  "max_similarity": 0.1382
}
```

---

## How It Works

### Data Preprocessing

Each title's metadata is combined into a single text `soup`:

```python
soup = title + director + cast + listed_in + description
```

All text is lowercased. Missing values are replaced with empty strings.

### CountVectorizer

Builds a sparse term-frequency matrix using scikit-learn's `CountVectorizer` with English stop-word removal. Each title becomes a vector of raw word counts. Similarity is computed via cosine similarity.

**Strength:** Fast, interpretable, good at matching exact keywords.  
**Weakness:** Treats all words equally regardless of how common they are.

### TF-IDF

Uses `TfidfVectorizer` to weight each term by how frequently it appears in a title (`TF`) divided by how common it is across all titles (`IDF`). Words like *"the"* or *"a"* are down-weighted; unique descriptive terms are up-weighted.

**Strength:** More discriminative and precise than raw counts.  
**Weakness:** Still purely lexical — no understanding of meaning.

### SBERT (Sentence-BERT)

Encodes each `soup` string into a 384-dimensional dense vector using the `all-MiniLM-L6-v2` transformer model with L2 normalization. Captures semantic meaning rather than exact word overlap.

**Strength:** Understands synonyms, context, and thematic similarity.  
**Weakness:** Slower to build; requires PyTorch.

### Similarity

All three models use **cosine similarity** to rank candidates:

```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```

Result scores are normalised relative to the top result in each query and displayed as a percentage bar in the UI.

---

## Dataset

**Source:** [Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows) — Kaggle  
**Size:** 8,807 titles (6,131 Movies + 2,676 TV Shows)  
**Fields used:** `title`, `director`, `cast`, `listed_in`, `description`

---

## Tech Stack

- **Backend:** Python, FastAPI, uvicorn
- **ML / NLP:** scikit-learn, sentence-transformers, numpy, scipy
- **Data:** pandas
- **Frontend:** Vanilla HTML, CSS, JavaScript (no framework dependencies)

---

## Research Notebook

The original analysis (`nlp_a3_Wed(Evening)Group17.ipynb`) covers:

- Model construction and timing benchmarks
- Pairwise cosine similarity distribution histograms for all three models
- Qualitative comparison of recommendation output per model
- Discussion of why SBERT's higher average similarity reflects semantic density, not noise

---

## License

MIT

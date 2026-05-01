# Pokedex SE

A Pokemon-themed search engine that crawls Pokemon wikis and fan sites, builds an inverted index, and ranks results using BM25 scoring combined with PageRank.

---

## Features

- **Web Crawler** — Scrapy-based spider that crawls Pokemon fandom sites up to depth 5
- **BM25 Ranking** — Industry-standard term frequency scoring with field weighting (title matches ranked higher than body text)
- **PageRank** — Link graph analysis via NetworkX to boost authoritative pages
- **Inverted Index** — Persistent index saved to disk so the server starts instantly on repeat runs
- **Minimal UI** — Flat, Pokemon-styled web interface served over localhost

---

## Tech Stack

| Library | Purpose |
|---|---|
| `scrapy` | Web crawling |
| `nltk` | Tokenisation, stemming, stop word removal |
| `networkx` | PageRank computation |
| Python `http.server` | Built-in web server |

---

## Setup

**Requirements:** Python 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Step 1 — Crawl (first time only)

```bash
py crawler.py
```

Crawls Pokemon wikis and fan sites and saves results to `outputsFile/crawled_data.json`. This takes several minutes.

### Step 2 — Start the search engine

```bash
py main.py
```

On first run, builds the inverted index from crawled data and saves it to `outputsFile/`. On subsequent runs, loads the saved index directly and starts immediately.

Open your browser at:

```
http://localhost:8000
```

---

## Usage

Type any Pokemon-related query into the search bar.

**Example queries:**
- `pikachu`
- `fire type moves`
- `legendary pokemon`
- `gym leaders kanto`
- `evolution stones`

---

## Project Structure

```
├── main.py               # Search engine, indexing, BM25 + PageRank, web server
├── crawler.py            # Scrapy spider and URL filtering logic
├── requirements.txt      # Python dependencies
├── instructions.txt      # Plain-text setup guide
├── background.png        # UI background image
├── Pokemon Solid.ttf     # Font used for headings and button
├── Pokemon Hollow.ttf    # Font used for search input
├── .gitignore
└── outputsFile/          # Generated at runtime, excluded from version control
    ├── crawled_data.json # Raw crawled page data
    ├── documents.json    # Serialised document store
    └── postings.json     # Serialised inverted index
```

---

## Re-indexing

To rebuild the index after a new crawl, delete the cached index files and restart:

```bash
del outputsFile\documents.json outputsFile\postings.json
py main.py
```

To start completely fresh (re-crawl everything):

```bash
rmdir /s /q outputsFile
py crawler.py
py main.py
```

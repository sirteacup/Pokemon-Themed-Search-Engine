# Pokedex SE

A Pokemon-themed search engine that crawls Pokemon wikis and fan sites, builds an inverted index, and ranks results using BM25 scoring combined with PageRank.

## Features

- Scrapy-based web crawler targeting Pokemon fandom sites
- BM25 ranking with field weighting (title matches score higher than body text)
- PageRank via NetworkX to boost authoritative pages
- Persistent index saved to disk for fast startup on repeat runs
- Minimal flat UI with Pokemon fonts served over localhost

## Project Structure

```
├── main.py               # Search engine, indexing, BM25 + PageRank, web server
├── crawler.py            # Scrapy spider and URL filtering
├── requirements.txt      # Python dependencies
├── instructions.txt      # Setup and usage guide
├── background.png        # UI background image
├── Pokemon Solid.ttf     # Font for headings and button
├── Pokemon Hollow.ttf    # Font for search input
└── outputsFile/          # Generated at runtime, excluded from version control
```

## Getting Started

See [instructions.txt](instructions.txt) for the full setup and usage guide.

Dependencies are listed in [requirements.txt](requirements.txt).

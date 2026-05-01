
import html
import os
import json
import math
import unicodedata
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, urlunparse, unquote
import networkx as nx
import nltk
from crawler import run_crawler

TITLE_BOOST = 3

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


# Normalize URL for matching: remove fragments, trailing slashes, and optionally query params
def normalize_url_for_matching(url):
    """Normalize a URL for comparison by removing fragments and trailing slashes."""
    if not url:
        return url
    parsed = urlparse(url)
    normalized = urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path.rstrip('/') if parsed.path != '/' else '/',
        parsed.params,
        '',
        ''
    ))
    return normalized



# Import NLTK components
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        
except ImportError:
    print("NLTK not installed.")
    exit(1)



# Path to index and crawl files inside outputsFile directory (same dir as this file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputsFile")
CRAWLED_DATA_FILE = os.path.join(OUTPUTS_DIR, "crawled_data.json")
DOCUMENTS_INDEX_FILE = os.path.join(OUTPUTS_DIR, "documents.json")
POSTINGS_INDEX_FILE = os.path.join(OUTPUTS_DIR, "postings.json")

# Main search engine class with indexing, retrieval, and PageRank.
class SearchEngine:
    
    def __init__(self):
        self.documents = {}  
        self.inverted_index = defaultdict(dict)  
        self.doc_lengths = {}  
        self.avg_doc_length = 0.0
        self.total_docs = 0
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.graph = nx.DiGraph()  # For PageRank
        self.pagerank_scores = {}  
        
    def tokenize_and_process(self, text):
        if not text:
            return []
        text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode()
        tokens = word_tokenize(text.lower())

        processed_tokens = []
        for token in tokens:
            if token.isalpha() and token not in self.stop_words:
                stemmed = self.stemmer.stem(token)
                if len(stemmed) > 1:  # Filter very short tokens
                    processed_tokens.append(stemmed)
        
        return processed_tokens
    
    def build_index(self, crawled_data):
        self.documents = {}
        self.inverted_index = defaultdict(dict)
        self.doc_lengths = {}
        self.graph = nx.DiGraph()
        
        # Store documents
        for doc_data in crawled_data:
            doc_id = doc_data['doc_id']
            self.documents[doc_id] = {
                'url': doc_data['url'],
                'title': doc_data['title'],
                'text': doc_data['text'],
                'links': doc_data.get('links', []) if 'links' in doc_data else []
            }
            
            # Index title and body separately; title terms weighted higher
            title_tokens = self.tokenize_and_process(doc_data['title'])
            body_tokens = self.tokenize_and_process(doc_data['text'])
            self.doc_lengths[doc_id] = len(title_tokens) + len(body_tokens)

            term_freq = defaultdict(int)
            for token in title_tokens:
                term_freq[token] += TITLE_BOOST
            for token in body_tokens:
                term_freq[token] += 1

            for term, freq in term_freq.items():
                self.inverted_index[term][doc_id] = freq
            
            # Build graph for PageRank 
            url = doc_data['url']
            self.graph.add_node(doc_id, url=url)
        
        url_to_doc_id = {}
        for doc_id, doc in self.documents.items():
            normalized_url = normalize_url_for_matching(doc['url'])
            url_to_doc_id[normalized_url] = doc_id
        
        for doc_id, doc in self.documents.items():
            for link in doc.get('links', []):
                normalized_link = normalize_url_for_matching(link)
                if normalized_link in url_to_doc_id:
                    target_doc_id = url_to_doc_id[normalized_link]
                    if doc_id != target_doc_id:
                        self.graph.add_edge(doc_id, target_doc_id)
        
        # Calculate average document length
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        
        self.total_docs = len(self.documents)
        
        # Compute PageRank
        self.compute_pagerank()

    def compute_pagerank(self):
        if len(self.graph.nodes()) == 0:
            self.pagerank_scores = {doc_id: 0.0 for doc_id in self.documents.keys()}
            return

        try:
            pr_scores = nx.pagerank(self.graph, alpha=0.85, max_iter=100)

            values = list(pr_scores.values())
            min_score = min(values)
            max_score = max(values)

            if max_score > min_score:
                self.pagerank_scores = {
                    doc_id: (score - min_score) / (max_score - min_score)
                    for doc_id, score in pr_scores.items()
                }
            else:
                self.pagerank_scores = {doc_id: 0.5 for doc_id in pr_scores.keys()}

        except Exception as e:
            self.pagerank_scores = {doc_id: 0.5 for doc_id in self.documents.keys()}

        for doc_id in self.documents.keys():
            if doc_id not in self.pagerank_scores:
                self.pagerank_scores[doc_id] = 0.0




# Compute BM25 score for a document given query terms.
    def bm25_score(self, query_terms, doc_id, k1=1.5, b=0.75):

        score = 0.0
        
        if doc_id not in self.documents:
            return 0.0
        
        doc_length = self.doc_lengths.get(doc_id, 0)
        if doc_length == 0 or self.avg_doc_length == 0:
            return 0.0
        
        for term in query_terms:
            if term in self.inverted_index and doc_id in self.inverted_index[term]:
                # Term frequency in document
                tf = self.inverted_index[term][doc_id]
                
                # Document frequency (number of documents containing term)
                df = len(self.inverted_index[term])
                
                idf = max(0.0, math.log((self.total_docs - df + 0.5) / (df + 0.5)))
                
                # BM25 component
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_length / self.avg_doc_length))
                
                score += idf * (numerator / denominator)
        
        return score

# Search documents using BM25 and PageRank.

    def search(self, query, top_k=10):
        if not query or self.total_docs == 0:
            return []
        
        # Process query
        query_terms = self.tokenize_and_process(query)
        if not query_terms:
            return []
        
        # Calculate BM25 scores, keeping only documents with at least one term match
        bm25_scores = {}
        for doc_id in self.documents.keys():
            bm25 = self.bm25_score(query_terms, doc_id)
            if bm25 > 0.0:
                bm25_scores[doc_id] = bm25

        if not bm25_scores:
            return []

        # Normalise BM25 to [0,1] among matching docs so it combines cleanly with PageRank
        max_bm25 = max(bm25_scores.values())
        bm25_scores = {doc_id: score / max_bm25 for doc_id, score in bm25_scores.items()}

        # Combine scores (only docs that had a BM25 match)
        combined_scores = {}
        for doc_id, bm25 in bm25_scores.items():
            pagerank = self.pagerank_scores.get(doc_id, 0.0)
            combined_scores[doc_id] = 0.8 * bm25 + 0.2 * pagerank
        
        # Sort by combined score
        ranked_docs = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top K results
        results = []
        for doc_id, score in ranked_docs[:top_k]:
            if score > 0:
                doc = self.documents[doc_id]
                results.append({
                    'doc_id': doc_id,
                    'url': doc['url'],
                    'title': doc['title'],
                    'text': doc['text'][:200] + '...' if len(doc['text']) > 200 else doc['text'],
                    'bm25_score': bm25_scores.get(doc_id, 0.0),
                    'pagerank_score': self.pagerank_scores.get(doc_id, 0.0),
                    'combined_score': score
                })
        
        return results



HOME_PAGE_HTML = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Pokedex SE</title>
    <style>
        @font-face {
            font-family: 'PokemonSolid';
            src: url('/Pokemon%20Solid.ttf') format('truetype');
        }
        @font-face {
            font-family: 'PokemonHollow';
            src: url('/Pokemon%20Hollow.ttf') format('truetype');
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            min-height: 100vh;
            background-image: url('/background.png');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }
        body::before {
            content: '';
            position: fixed;
            inset: 0;
            background: rgba(10, 20, 60, 0.55);
            z-index: 0;
        }
        .container {
            position: relative;
            z-index: 1;
            background: #fff;
            width: 100%;
            max-width: 560px;
            padding: 48px 40px;
        }
        h1 {
            font-family: 'PokemonSolid', sans-serif;
            text-align: center;
            font-size: 2.2rem;
            color: #1d4ed8;
            -webkit-text-stroke: 2px #FFD700;
            margin-bottom: 32px;
            letter-spacing: 1px;
        }
        .search-box {
            display: flex;
            border: 1px solid #d1d5db;
            overflow: hidden;
        }
        input[type="text"] {
            flex: 1;
            padding: 12px 16px;
            font-size: 14px;
            border: none;
            outline: none;
            color: #111;
            background: #fff;
        }
        input[type="text"]::placeholder { color: #9ca3af; }
        button {
            padding: 12px 22px;
            font-size: 14px;
            font-weight: 700;
            font-family: 'PokemonSolid', sans-serif;
            background: #FFD700;
            color: #1d4ed8;
            border: none;
            cursor: pointer;
            white-space: nowrap;
        }
        button:hover { background: #f0c800; }
        .info {
            margin-top: 16px;
            font-size: 12px;
            color: #6b7280;
            border-left: 2px solid #e5e7eb;
            padding-left: 12px;
        }
        .stats {
            text-align: center;
            color: #9ca3af;
            margin-top: 20px;
            font-size: 11px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Pokedex SE</h1>
        <form method="GET" action="/search">
            <div class="search-box">
                <input type="text" name="q" placeholder="Search Pokemon, characters, items, locations..." required>
                <button type="submit">Search</button>
            </div>
        </form>
        <div class="info">Try: Pokemon names, characters, items, locations</div>
        <div class="stats">{doc_count} documents indexed</div>
    </div>
</body>
</html>"""

RESULTS_PAGE_HTML = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{query} | Pokedex SE</title>
    <style>
        @font-face {
            font-family: 'PokemonSolid';
            src: url('/Pokemon%20Solid.ttf') format('truetype');
        }
        @font-face {
            font-family: 'PokemonHollow';
            src: url('/Pokemon%20Hollow.ttf') format('truetype');
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            min-height: 100vh;
            background-image: url('/background.png');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            position: relative;
        }
        body::before {
            content: '';
            position: fixed;
            inset: 0;
            background: rgba(10, 20, 60, 0.55);
            z-index: 0;
        }
        .header {
            position: relative;
            z-index: 1;
            background: #fff;
            border-bottom: 1px solid #e5e7eb;
            padding: 14px 32px;
            display: flex;
            align-items: center;
            gap: 24px;
        }
        .header a {
            font-family: 'PokemonSolid', sans-serif;
            font-size: 1.1rem;
            color: #1d4ed8;
            -webkit-text-stroke: 1.5px #FFD700;
            text-decoration: none;
            white-space: nowrap;
        }
        .search-box {
            display: flex;
            border: 1px solid #d1d5db;
            flex: 1;
            max-width: 560px;
            overflow: hidden;
        }
        input[type="text"] {
            flex: 1;
            padding: 9px 14px;
            font-size: 14px;
            border: none;
            outline: none;
            background: #fff;
            color: #111;
        }
        input[type="text"]::placeholder { color: #9ca3af; }
        button {
            padding: 9px 18px;
            font-size: 13px;
            font-weight: 700;
            font-family: 'PokemonSolid', sans-serif;
            background: #FFD700;
            color: #1d4ed8;
            border: none;
            cursor: pointer;
        }
        button:hover { background: #f0c800; }
        .main {
            position: relative;
            z-index: 1;
            max-width: 740px;
            margin: 28px auto;
            padding: 0 24px 48px;
        }
        .results-box {
            background: #fff;
            padding: 24px 28px;
        }
        .query-info {
            font-size: 13px;
            color: #6b7280;
            margin-bottom: 16px;
            padding-bottom: 14px;
            border-bottom: 1px solid #f3f4f6;
        }
        .result-item {
            padding: 16px 0;
            border-bottom: 1px solid #f3f4f6;
        }
        .result-item:last-child { border-bottom: none; }
        .result-title {
            font-size: 16px;
            font-weight: 600;
            color: #1d4ed8;
            text-decoration: none;
            display: block;
            margin-bottom: 3px;
        }
        .result-title:hover { text-decoration: underline; }
        .result-url {
            font-size: 12px;
            color: #16a34a;
            margin-bottom: 6px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .result-text {
            font-size: 13px;
            color: #4b5563;
            line-height: 1.6;
        }
        .result-scores {
            margin-top: 6px;
            font-size: 11px;
            color: #9ca3af;
        }
        .no-results {
            padding: 48px 0;
            font-size: 15px;
            color: #6b7280;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="header">
        <a href="/">Pokedex SE</a>
        <form method="GET" action="/search">
            <div class="search-box">
                <input type="text" name="q" value="{query}" placeholder="Search..." required>
                <button type="submit">Search</button>
            </div>
        </form>
    </div>
    <div class="main">
        <div class="results-box">
            <div class="query-info">{result_count} results for &ldquo;{query}&rdquo;</div>
            <div class="results">{results_html}</div>
        </div>
    </div>
</body>
</html>"""

# HTTP request handler for the search engine web interface.
class SearchEngineHandler(BaseHTTPRequestHandler):
    
    def __init__(self, *args, search_engine=None, **kwargs):
        self.search_engine = search_engine
        super().__init__(*args, **kwargs)
# Handle GET requests.    
    def do_GET(self):

        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/' or parsed_path.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            
            doc_count = self.search_engine.total_docs if self.search_engine else 0
            html_content = HOME_PAGE_HTML.replace('{doc_count}', str(doc_count))
            self.wfile.write(html_content.encode('utf-8'))
            
        elif parsed_path.path == '/search':
            query_params = parse_qs(parsed_path.query)
            query = query_params.get('q', [''])[0]
            
            results = []
            if query and self.search_engine:
                results = self.search_engine.search(query)
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            
            # Build results HTML
            if results:
                results_html = ''
                for result in results:
                    # Escape HTML entities for safety
                    escaped_url = html.escape(result['url'])
                    escaped_title = html.escape(result['title'] or 'No Title')
                    escaped_text = html.escape(result['text'])
                    results_html += f'''
                    <div class="result-item">
                        <a href="{escaped_url}" class="result-title" target="_blank">{escaped_title}</a>
                        <div class="result-url">{escaped_url}</div>
                        <div class="result-text">{escaped_text}</div>
                        <div class="result-scores">
                            Combined Score: {result['combined_score']:.4f} | 
                            BM25: {result['bm25_score']:.4f} | 
                            PageRank: {result['pagerank_score']:.4f}
                        </div>
                    </div>
                    '''
            else:
                results_html = '<div class="no-results">No results found.</div>'
            
            # Escape query for HTML safety
            escaped_query = html.escape(query)
            html_content = RESULTS_PAGE_HTML.replace('{query}', escaped_query)
            html_content = html_content.replace('{result_count}', str(len(results)))
            html_content = html_content.replace('{results_html}', results_html)
            self.wfile.write(html_content.encode('utf-8'))
            
        else:
            filename = unquote(parsed_path.path.lstrip('/'))
            filepath = os.path.join(BASE_DIR, filename)
            ext = os.path.splitext(filename)[1].lower()
            mime = {'.png': 'image/png', '.jpg': 'image/jpeg',
                    '.ttf': 'font/ttf', '.otf': 'font/otf', '.woff': 'font/woff', '.woff2': 'font/woff2'}
            if ext in mime and os.path.isfile(filepath):
                with open(filepath, 'rb') as f:
                    data = f.read()
                self.send_response(200)
                self.send_header('Content-type', mime[ext])
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'404 Not Found')
    
    def log_message(self, format, *args):
        pass

# function to create handler with search engine instance.
def make_handler(search_engine):
    def handler(*args, **kwargs):
        return SearchEngineHandler(*args, search_engine=search_engine, **kwargs)
    return handler

# Save documents and postings index to JSON files in outputsFile.
def save_index_to_disk(engine):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    documents_serializable = {
        str(doc_id): doc for doc_id, doc in engine.documents.items()
    }

    postings_serializable = {}
    for term, postings in engine.inverted_index.items():
        postings_serializable[term] = {
            str(doc_id): freq for doc_id, freq in postings.items()
        }

    with open(DOCUMENTS_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(documents_serializable, f, indent=2, ensure_ascii=False)

    with open(POSTINGS_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(postings_serializable, f, indent=2, ensure_ascii=False)

# MAIN
def main():
    print("Pokemon Search Engine Initiated.")
    print("=" * 60)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    engine = SearchEngine()

    index_exists = os.path.exists(DOCUMENTS_INDEX_FILE) and os.path.exists(POSTINGS_INDEX_FILE)

    if index_exists:
        # Index already built - load it directly, never touch crawled_data.json
        try:
            with open(DOCUMENTS_INDEX_FILE, "r", encoding="utf-8") as f:
                documents_data = json.load(f)
            with open(POSTINGS_INDEX_FILE, "r", encoding="utf-8") as f:
                postings_data = json.load(f)

            engine.documents = {
                int(doc_id): doc for doc_id, doc in documents_data.items()
            }

            engine.inverted_index = defaultdict(dict)
            for term, postings in postings_data.items():
                engine.inverted_index[term] = {
                    int(doc_id): freq for doc_id, freq in postings.items()
                }

            engine.doc_lengths = {}
            engine.graph = nx.DiGraph()

            for doc_id, doc in engine.documents.items():
                title_tokens = engine.tokenize_and_process(doc['title'])
                body_tokens = engine.tokenize_and_process(doc['text'])
                engine.doc_lengths[doc_id] = len(title_tokens) + len(body_tokens)
                engine.graph.add_node(doc_id, url=doc['url'])

            url_to_doc_id = {}
            for doc_id, doc in engine.documents.items():
                normalized_url = normalize_url_for_matching(doc['url'])
                url_to_doc_id[normalized_url] = doc_id

            for doc_id, doc in engine.documents.items():
                for link in doc.get('links', []):
                    normalized_link = normalize_url_for_matching(link)
                    if normalized_link in url_to_doc_id:
                        target_doc_id = url_to_doc_id[normalized_link]
                        if doc_id != target_doc_id:
                            engine.graph.add_edge(doc_id, target_doc_id)

            if engine.doc_lengths:
                engine.avg_doc_length = sum(engine.doc_lengths.values()) / len(engine.doc_lengths)
            engine.total_docs = len(engine.documents)

            engine.compute_pagerank()
            print(f"✓ Loaded existing index ({engine.total_docs} documents, {len(engine.inverted_index)} terms)")

        except Exception as e:
            print(f"✗ Failed to load index: {e}")
            print("  Delete outputsFile/documents.json and outputsFile/postings.json, then run again to rebuild.")
            return

    else:
        # No index - build from crawled_data.json, running the crawler first if needed
        if not os.path.exists(CRAWLED_DATA_FILE):
            print(f"✗ No crawled data found. Running crawler...")
            try:
                run_crawler(CRAWLED_DATA_FILE)
            except Exception as e:
                print(f"✗ Crawler error: {e}")
                return

        try:
            with open(CRAWLED_DATA_FILE, 'r', encoding='utf-8') as f:
                crawled_data = json.load(f)
            print(f"Loaded {len(crawled_data)} documents from crawled data...")
        except FileNotFoundError:
            print("✗ crawled_data.json was not created. Run: py crawler.py")
            return
        except MemoryError:
            print("✗ crawled_data.json is too large to load into memory.")
            print("  Run: py crawler.py  to generate a fresh crawl, then try again.")
            return

        if not crawled_data:
            print("✗ No crawled data. Run: py crawler.py")
            return

        print("Building index...")
        engine.build_index(crawled_data)
        print(f"✓ Index built - {engine.total_docs} documents, {len(engine.inverted_index)} unique terms")
        print(f"  Average document length: {engine.avg_doc_length:.2f}")
        save_index_to_disk(engine)
        print("✓ Index saved to outputsFile/")


    # Always save the (possibly updated) index to disk
    save_index_to_disk(engine)

    
    # Start web server
    port = 8000
    handler = make_handler(engine)
    httpd = HTTPServer(('localhost', port), handler)
    
    print("=" * 60)
    print(f"Web interface available at: http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()


if __name__ == '__main__':
    main()


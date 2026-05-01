
import os
import logging
import scrapy
from scrapy.crawler import CrawlerProcess
import json
import re
import unicodedata


# Get the directory where this script is located
BASE_DIR = os.path.dirname(__file__)

# Define the output directory as a subdirectory called "outputsFile" in the script's directory
OUTPUT_DIR = os.path.join(BASE_DIR, "outputsFile")

# Default path for the crawled data JSON file inside the outputsFile directory
DEFAULT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "crawled_data.json")

# Module-level list to store crawled data that can be accessed after crawler execution
_crawled_data_storage = []



# Normalize Unicode text to ASCII-compatible lowercase for keyword matching.
def normalize(text):
    return unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode().lower()



# URL FILTERING CONFIGURATION

# List of allowed domain names that the crawler is permitted to visit

ALLOWED_DOMAINS = (
    "pokemon.fandom.com",
    "bulbapedia.bulbagarden.net",
    "www.pokemon.com",
    "pokemon.com",
    "www.serebii.net",
    "serebii.net",
    "www.reddit.com",
    "reddit.com",
)

# List of Wikipedia domains in other languages that should be blocked to focus on english content. 
# This is leftover from when wikipedia was being used as a seed url and I am afraid of removing it.
OTHER_LANGUAGE_WIKIS = (
    "https://de.wikipedia.org/",
    "https://fr.wikipedia.org/",
    "https://es.wikipedia.org/",
    "https://it.wikipedia.org/",
    "https://ja.wikipedia.org/",
    "https://ru.wikipedia.org/",
    "https://zh.wikipedia.org/",
)


FORBIDDEN_SUBSTRINGS = (
    "wikidata.org",
    "/w/index.php",
    "action=edit",
    "veaction=edit",
    "&oldid=",
    "?oldid=",
    "Special:",
    "Talk:",
    "Category:",
    "Template:",
    "Help:",
    "Wikipedia:",
)

# E-commerce and app store domains that should be blocked

FORBIDDEN_STORE_SITES = (
    "play.google.com",
    "apps.apple.com",
    "itunes.apple.com",
    "amazon.com",
    "amazon.co",
    "bestbuy.com",
    "ebay.com",
)

# Social media and social networking sites that should be blocked

FORBIDDEN_SOCIAL_SITES = (
    "facebook.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "tiktok.com",
    "tumblr.com",
    "linkedin.com",
    "pinterest.com",
    "snapchat.com", 
    "https://bulbagarden.tumblr.com", 
    "https://thehumanwiki.tumblr.com", 
    "https://beakit.tumblr.com", 
    "https://eggtempest.tumblr.com/" 
)


# Used to block non-English language variants of pages
LANG_SUFFIX_RE = re.compile(r"_\([a-z]{2}\)$", re.IGNORECASE)



# Determine whether a URL should be followed (crawled) based on various filtering rules.

def should_follow(url: str) -> bool:

    # Convert URL to lowercase  
    url_l = url.lower()

    # Check if URL ends with a file extension that indicates binary/non-HTML content
    bad_ext = (".zip",".tar",".gz",".bz2",".pdf",".jpg",".jpeg",".png",".gif",".svg",".exe",".msi",".whl",".iso")
    if url_l.endswith(bad_ext):
        return False

    # Block URLs from non-English Wikipedia domains
    if any(url.startswith(domain) for domain in OTHER_LANGUAGE_WIKIS):
        return False

    # Block URLs with language suffixes 
    if LANG_SUFFIX_RE.search(url):
        return False

    # Block URLs containing forbidden substrings
    if any(substr in url for substr in FORBIDDEN_SUBSTRINGS):
        return False

    # Block URLs from e-commerce and app store sites
    if any(store in url_l for store in FORBIDDEN_STORE_SITES):
        return False

    # Block URLs from social media and social networking sites
    if any(social in url_l for social in FORBIDDEN_SOCIAL_SITES):
        return False

    # Whitelist check: Only allow URLs from approved domains
    if not any(
        url_l.startswith("https://" + domain) or url_l.startswith("http://" + domain)
        for domain in ALLOWED_DOMAINS
    ):
        return False

    # If all checks pass, the URL is allowed
    return True

# SCRAPY SPIDER CLASS

# Scrapy spider that crawls Pokemon-related web pages.
class SearchEngineSpider(scrapy.Spider):
 
    # Spider name identifier used by Scrapy
    name = 'search_engine'

    start_urls = [
        'https://pokemon.fandom.com/wiki/Pok%C3%A9mon_Wiki',
        'https://bulbapedia.bulbagarden.net/wiki/Main_Page',
        'https://www.pokemon.com/us',
        'https://www.reddit.com/r/pokemon/',
        'https://pokemon.fandom.com/wiki/Games',
        'https://www.serebii.net/'
    ]

    custom_settings = {

        'ROBOTSTXT_OBEY': True,
        'CONCURRENT_REQUESTS': 16,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 8,
        'DOWNLOAD_DELAY': 0.1,
        'RANDOMIZE_DOWNLOAD_DELAY': False,
        'DEPTH_LIMIT': 5,
        'DOWNLOAD_TIMEOUT': 10,
        'LOG_LEVEL': 'INFO',
        'HTTPCACHE_ENABLED': True,
        'HTTPCACHE_EXPIRATION_SECS': 60 * 60 * 24,  # 24 hours
        'HTTPCACHE_DIR': 'httpcache',  # Directory where cache is stored
        'HTTPCACHE_IGNORE_HTTP_CODES': [404],  # Don't cache 404 errors
    }

#Initialize the spider instance. Sets up instance variables for tracking crawled data and configuration. The output_file can be overridden via kwargs when the spider is instantiated.

    def __init__(self, *args, **kwargs):

        # Call parent class constructor
        super().__init__(*args, **kwargs)
        
        # Counter for assigning unique document IDs to each indexed page
        self.doc_id = 0   
        # Set to track URLs that have already been visited to avoid duplicates
        self.visited = set()
        
        # List to store document data for all indexed pages
        # Each element is a dictionary with doc_id, url, title, and text
        self.crawled_data = []
        
        # Output file path for saving crawled data

        self.output_file = kwargs.get('output_file', DEFAULT_OUTPUT_PATH)

#  Parse a downloaded web page response. Does duplicate detection, extracts title and text content, checks for keyword pokemon, indexes the page if keyword is found
    def parse(self, response):

        if response.url in self.visited:
            return
        
        # Mark the URL as visited before processing
        self.visited.add(response.url)

        # Extract the page title 
        title = (response.xpath("//title/text()").get() or "").strip()

        # Extract visible text content from the page
        text_parts = response.xpath("//p//text() | //h1//text() | //h2//text()").getall()
        

        visible_text = " ".join(t.strip() for t in text_parts if t.strip())
        visible_text = " ".join(visible_text.split())
        normalized_text = normalize(visible_text)

        # Keyword filter: page must mention "pokemon" in text or title
        normalized_title = normalize(title)
        if "pokemon" not in normalized_text and "pokemon" not in normalized_title:
            return


        self.doc_id += 1

        # Single-pass link extraction: filter, deduplicate, and optionally yield requests
        depth = response.meta.get("depth", 0)
        seen_links = set()
        normalized_links = []
        for link in response.xpath("//a/@href").getall():
            absolute = response.urljoin(link).split("#")[0]
            if not should_follow(absolute) or absolute in seen_links:
                continue
            seen_links.add(absolute)
            normalized_links.append(absolute)
            if depth < self.custom_settings["DEPTH_LIMIT"] and absolute not in self.visited:
                yield scrapy.Request(
                    absolute,
                    callback=self.parse,
                    meta={"depth": depth + 1},
                    dont_filter=True,
                )

        # Create a dictionary containing all the data for this document
        doc = {
            "doc_id": self.doc_id,
            "url": response.url,
            "title": title,
            "text": visible_text,
            "links": normalized_links
        }

        # Add this document to the list of crawled data
        self.crawled_data.append(doc)
# Called when the spider finishes crawling (either normally or due to an error).
    def closed(self, reason):

        # Update the global storage variable so it can be accessed after crawler execution
        global _crawled_data_storage
        _crawled_data_storage = self.crawled_data

        try:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(self.crawled_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Saving failed: {e}")



# CRAWLER EXECUTION FUNCTION
# Execute the web crawler and return the crawled data.

def run_crawler(output_file=None):

    global _crawled_data_storage    
    _crawled_data_storage = []

    if output_file is None:
        output_file = DEFAULT_OUTPUT_PATH


    process = CrawlerProcess()
    logging.getLogger('scrapy.spidermiddlewares.httperror').setLevel(logging.WARNING)
    process.crawl(SearchEngineSpider, output_file=output_file)
    process.start()
    return _crawled_data_storage


# MAIN

if __name__ == "__main__":

    data = run_crawler()
    print(f"Crawled {len(data)} pages")

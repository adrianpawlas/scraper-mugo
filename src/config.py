"""Configuration and environment variables for the scraper."""

import os
from dotenv import load_dotenv

load_dotenv()


# --- Supabase ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError(
        "SUPABASE_URL and SUPABASE_KEY must be set in environment or .env file"
    )

# --- Scraper ---
BASE_URL = os.getenv("BASE_URL", "https://www.mugodepartment.com")
COLLECTION_URL = os.getenv(
    "COLLECTION_URL", "https://www.mugodepartment.com/collections/all"
)
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", "0.5"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
SHOPIFY_PAGE_LIMIT = 250

# --- Brand ---
SOURCE = "scraper-mugo"
BRAND = "Mugo Department"
SECOND_HAND = False

# --- Embeddings ---
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "google/siglip-base-patch16-384"
)
EMBEDDING_DIM = 768

# --- Paths ---
# Expand tilde so os.path operations work correctly (os.path.isdir, shutil.rmtree, etc.)
CACHE_DIR = os.path.expanduser(os.getenv("CACHE_DIR", "./.cache"))

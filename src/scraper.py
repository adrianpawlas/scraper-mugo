"""Shopify product scraper for Mugo Department.

Fetches all products from the Shopify JSON API, extracts fields,
and generates embeddings only for new or changed products.
"""

import json
import logging
import re
import time
from datetime import datetime, timezone
from html import unescape
from typing import Any, Optional

import requests
from tqdm import tqdm

from src.config import (
    BASE_URL,
    BRAND,
    MAX_RETRIES,
    RATE_LIMIT_DELAY,
    SECOND_HAND,
    SHOPIFY_PAGE_LIMIT,
    SOURCE,
)
from src.embeddings import get_image_embedding, get_text_embedding

logger = logging.getLogger(__name__)

# Session for connection reuse
_session = requests.Session()
_session.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (compatible; MugoScraper/1.0; "
            "+https://github.com/adrianpawlas/scraper-mugo)"
        ),
        "Accept": "application/json",
    }
)

# Embedding delay between API calls to avoid overwhelming the endpoint
_EMBEDDING_DELAY = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _request(url: str, retries: int = MAX_RETRIES) -> Optional[dict[str, Any]]:
    """Make a GET request with retries and rate-limiting."""
    for attempt in range(1, retries + 1):
        try:
            resp = _session.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.warning(
                "Request failed (attempt %d/%d): %s — %s",
                attempt,
                retries,
                url,
                exc,
            )
            if attempt < retries:
                time.sleep(RATE_LIMIT_DELAY * attempt)
    return None


def _clean_html(html_text: str) -> str:
    """Strip HTML tags and unescape entities."""
    text = re.sub(r"<[^>]+>", " ", html_text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_product_id(variants: list[dict]) -> str:
    """Build a stable product ID from the first variant's ID."""
    return str(variants[0]["id"]) if variants else "unknown"


def _build_product_url(handle: str) -> str:
    return f"{BASE_URL}/products/{handle}"


def _extract_price_info(
    variants: list[dict],
) -> tuple[Optional[str], Optional[str]]:
    """Extract price and sale from the first variant."""
    if not variants:
        return None, None

    v = variants[0]
    raw_price = v.get("price")
    raw_compare = v.get("compare_at_price")
    currency = v.get("price_currency", "EUR")

    price = None
    sale = None

    if raw_price:
        price = f"{raw_price}{currency}"

    if raw_compare and raw_price:
        try:
            cp = float(raw_compare.replace(",", "."))
            p = float(raw_price.replace(",", "."))
            if cp > p:
                sale = f"{raw_price}{currency}"
                price = f"{raw_compare}{currency}"
        except (ValueError, AttributeError):
            pass

    currencies_found: set[str] = set()
    all_prices: list[str] = []
    for var in variants:
        var_price = var.get("price")
        var_currency = var.get("price_currency", "EUR")
        if var_price:
            key = f"{var_price}{var_currency}"
            if key not in currencies_found:
                currencies_found.add(key)
                all_prices.append(key)

    if len(all_prices) > 1:
        price = " , ".join(all_prices)
        if sale:
            sale = f"{raw_price}{currency}"

    return price, sale


def _extract_size(variants: list[dict], options: list[dict]) -> Optional[str]:
    """Extract sizes from variant options."""
    sizes = set()
    for v in variants:
        for key, val in v.items():
            if key.startswith("option") and val and val.strip():
                idx = int(key.replace("option", "")) - 1
                if idx < len(options):
                    opt_name = options[idx].get("name", "").lower()
                    if "size" in opt_name:
                        sizes.add(val.strip())
    return ", ".join(sorted(sizes)) if sizes else None


def _extract_category(product_type: str, tags: list[str]) -> Optional[str]:
    """Build category from product type and relevant tags."""
    categories = []

    if product_type:
        parts = re.split(r"\s*[&,/]\s*", product_type)
        categories.extend(p.strip() for p in parts if p.strip())

    category_keywords = {
        "sweater", "hoodie", "tee", "t-shirt", "pants", "jacket",
        "coat", "shirt", "jeans", "shorts", "hat", "cap", "beanie",
        "bag", "backpack", "accessories", "shoes", "socks", "scarf",
        "gloves", "belt", "jumper", "cardigan", "vest",
    }
    for tag in tags:
        tl = tag.lower().strip()
        if tl in category_keywords and tl not in (c.lower() for c in categories):
            categories.append(tag)

    return ", ".join(categories) if categories else None


def _build_metadata(
    product: dict[str, Any],
    price: Optional[str],
    sale: Optional[str],
    size: Optional[str],
    category: Optional[str],
    description: str,
) -> str:
    """Build a comprehensive metadata JSON string."""
    meta = {
        "title": product.get("title"),
        "description": description,
        "vendor": product.get("vendor"),
        "product_type": product.get("product_type"),
        "category": category,
        "price": price,
        "sale": sale,
        "size": size,
        "tags": product.get("tags", []),
        "options": product.get("options"),
        "variants_count": len(product.get("variants", [])),
        "images_count": len(product.get("images", [])),
        "handle": product.get("handle"),
        "created_at_shopify": product.get("created_at"),
        "updated_at_shopify": product.get("updated_at"),
        "missed_runs": 0,
    }
    return json.dumps(meta, ensure_ascii=False)


def _build_info_text(
    title: str,
    description: str,
    category: Optional[str],
    gender: Optional[str],
    price: Optional[str],
    sale: Optional[str],
    size: Optional[str],
    tags: list[str],
    brand: str,
) -> str:
    """Build a rich text string for the info_embedding."""
    parts = [
        f"Title: {title}",
        f"Brand: {brand}",
        f"Category: {category}" if category else None,
        f"Gender: {gender}" if gender else None,
        f"Price: {price}" if price else None,
        f"Sale: {sale}" if sale else None,
        f"Size: {size}" if size else None,
        f"Tags: {', '.join(tags)}" if tags else None,
        f"Description: {description}" if description else None,
    ]
    return "\n".join(p for p in parts if p)


def _normalise_tags(tags_val: Any) -> Optional[str]:
    """Normalize a tags value to a sorted comma-separated string for comparison.

    Handles lists (jsonb), serialised JSON arrays, and raw strings.
    """
    if tags_val is None:
        return None
    if isinstance(tags_val, list):
        return ", ".join(sorted(str(t).strip() for t in tags_val if str(t).strip()))
    if isinstance(tags_val, str):
        try:
            parsed = json.loads(tags_val)
            if isinstance(parsed, list):
                return ", ".join(sorted(str(t).strip() for t in parsed if str(t).strip()))
        except (json.JSONDecodeError, TypeError):
            pass
        # Fallback: treat as comma-separated string
        parts = [t.strip() for t in tags_val.split(",") if t.strip()]
        return ", ".join(sorted(parts)) if parts else None
    return str(tags_val)


def _compare_fields(
    scraped: dict[str, Any],
    existing: dict[str, Any],
    fields: list[str],
) -> bool:
    """Check if any of the given fields differ between scraped and existing records.

    Returns True if a change was detected.
    """
    for field in fields:
        old = existing.get(field)
        new = scraped.get(field)
        # Normalize None and empty string to the same value
        if not old and not new:
            continue
        if old != new:
            return True
    return False


# ---------------------------------------------------------------------------
# Main scraper
# ---------------------------------------------------------------------------

def fetch_all_product_handles() -> list[dict[str, Any]]:
    """Fetch all products from the Shopify collection JSON API with pagination.

    Returns raw product dicts (with basic variant/image data).
    """
    products_raw: list[dict[str, Any]] = []
    page = 1

    collection_json = _request(f"{BASE_URL}/collections/all.json")
    total_products = 0
    if collection_json:
        total_products = collection_json.get("collection", {}).get("products_count", 0)
        logger.info("Total products in collection: %d", total_products)

    logger.info("Fetching product list from Shopify JSON API...")
    pbar = tqdm(desc="Fetching product pages", unit="page")

    while True:
        url = (
            f"{BASE_URL}/collections/all/products.json"
            f"?limit={SHOPIFY_PAGE_LIMIT}&page={page}"
        )
        data = _request(url)
        if not data:
            break

        products = data.get("products", [])
        if not products:
            break

        products_raw.extend(products)
        pbar.update(1)
        page += 1
        time.sleep(RATE_LIMIT_DELAY)

    pbar.close()
    logger.info("Found %d products total", len(products_raw))
    return products_raw


def scrape_product_detail(handle: str) -> Optional[dict[str, Any]]:
    """Fetch the full detail JSON for a single product by handle."""
    url = f"{BASE_URL}/products/{handle}.json"
    data = _request(url)
    if data and "product" in data:
        return data["product"]
    return None


def process_product(
    product: dict[str, Any],
    existing_record: Optional[dict[str, Any]] = None,
) -> tuple[Optional[dict[str, Any]], str]:
    """Process a single product dict into a database-ready record.

    Compares against existing_record (if provided) to determine if the
    product is new, changed, or unchanged.  Embeddings are only generated
    for new products or when the image URL changes.

    Returns (record or None, action) where action is one of:
        'new'      – product doesn't exist in DB yet
        'updated'  – product changed (scraped data differs from existing)
        'skipped'  – product unchanged, no action needed
    """
    handle = product.get("handle", "")
    title = product.get("title", "").strip()
    if not title:
        logger.warning("Skipping product with no title (handle=%s)", handle)
        return None, "new"

    # --- Core fields ---
    product_type = product.get("product_type", "") or ""
    tags = product.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    variants = product.get("variants", [])
    images = product.get("images", [])
    body_html = product.get("body_html", "") or ""

    description = _clean_html(body_html)
    product_url = _build_product_url(handle)
    price, sale = _extract_price_info(variants)
    size = _extract_size(variants, product.get("options", []))
    category = _extract_category(product_type, tags)

    # --- Main image ---
    image_url = None
    if images:
        image_url = images[0].get("src")
        if not image_url and "url" in images[0]:
            image_url = images[0]["url"]
        if not image_url and "original_src" in images[0]:
            image_url = images[0]["original_src"]

    if not image_url:
        logger.warning("No image found for %s, skipping embedding", title)

    # --- Additional images ---
    additional_images = None
    if len(images) > 1:
        urls = []
        for img in images[1:]:
            src = img.get("src") or img.get("url") or img.get("original_src")
            if src:
                urls.append(src)
        if urls:
            additional_images = " , ".join(urls)

    # --- Build scraped comparison data ---
    # Normalise tags to sorted comma-separated string for robust comparison
    # regardless of how the DB stores them (jsonb list, TEXT, etc.)
    normalised_tags = _normalise_tags(tags)
    scraped_data = {
        "title": title,
        "price": price,
        "sale": sale,
        "description": description,
        "image_url": image_url,
        "additional_images": additional_images,
        "size": size,
        "category": category,
        "tags": normalised_tags,
    }

    # --- Decide action based on comparison with existing record ---
    action = "new"
    changed_fields = [
        "title", "price", "sale", "description",
        "image_url", "additional_images", "size",
        "category", "tags",
    ]

    if existing_record:
        # Check if data changed
        has_changes = _compare_fields(scraped_data, existing_record, changed_fields)
        if not has_changes:
            has_changes = existing_record.get("title") != title

        if has_changes:
            action = "updated"
        else:
            action = "skipped"

    # --- Embeddings (only for new products or when image changes) ---
    image_embedding = None
    info_embedding = None

    regenerate_embeddings = action == "new" or (
        action == "updated" and image_url != existing_record.get("image_url")
    )

    if regenerate_embeddings and image_url:
        image_embedding = get_image_embedding(image_url)
        time.sleep(_EMBEDDING_DELAY)

        info_text = _build_info_text(
            title=title,
            description=description,
            category=category,
            gender=None,
            price=price,
            sale=sale,
            size=size,
            tags=tags,
            brand=BRAND,
        )
        if info_text.strip():
            info_embedding = get_text_embedding(info_text)
            time.sleep(_EMBEDDING_DELAY)
    elif action == "updated" and not regenerate_embeddings:
        # Data changed but image is the same – carry over old embeddings
        image_embedding = existing_record.get("image_embedding")
        info_embedding = existing_record.get("info_embedding")

    # --- Metadata ---
    metadata = _build_metadata(
        product=product,
        price=price,
        sale=sale,
        size=size,
        category=category,
        description=description,
    )

    # --- Build the record ---
    record: dict[str, Any] = {
        "id": _extract_product_id(variants),
        "source": SOURCE,
        "product_url": product_url,
        "affiliate_url": None,
        "image_url": image_url,
        "brand": BRAND,
        "title": title,
        "description": description or None,
        "category": category,
        "gender": None,
        "image_embedding": image_embedding,
        "info_embedding": info_embedding,
        "price": price,
        "sale": sale,
        "second_hand": SECOND_HAND,
        "size": size,
        "additional_images": additional_images,
        "metadata": metadata,
        "tags": tags if tags else None,
        "country": None,
        "compressed_image_url": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    return record, action


# ---------------------------------------------------------------------------
# Scraper entry point
# ---------------------------------------------------------------------------

ScraperResult = tuple[
    list[dict[str, Any]],  # records to upsert (new + updated)
    int,                   # new count
    int,                   # updated count
    int,                   # skipped count
    set[str],              # seen product_urls
]


def run_scraper(
    existing_map: Optional[dict[str, dict[str, Any]]] = None,
) -> ScraperResult:
    """Run the full scraper pipeline.

    Args:
        existing_map: Dict of existing products keyed by product_url
                      (from supabase_client.fetch_existing_products).

    Returns:
        (records_to_upsert, new_count, updated_count, skipped_count, seen_urls)
    """
    if existing_map is None:
        existing_map = {}

    products_raw = fetch_all_product_handles()

    if not products_raw:
        logger.warning("No products found to scrape!")
        return [], 0, 0, 0, set()

    records_to_upsert: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    new_count = 0
    updated_count = 0
    skipped_count = 0

    for product in tqdm(products_raw, desc="Processing products", unit="product"):
        product_url = _build_product_url(product.get("handle", ""))
        existing_record = existing_map.get(product_url)

        record, action = process_product(product, existing_record)

        if record:
            seen_urls.add(record["product_url"])

            if action == "skipped":
                skipped_count += 1
            else:
                records_to_upsert.append(record)
                if action == "new":
                    new_count += 1
                else:
                    updated_count += 1

        time.sleep(RATE_LIMIT_DELAY)

    logger.info(
        "Scraping complete: %d new, %d updated, %d skipped (of %d total)",
        new_count,
        updated_count,
        skipped_count,
        len(products_raw),
    )
    return records_to_upsert, new_count, updated_count, skipped_count, seen_urls

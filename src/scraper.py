"""Shopify product scraper for Mugo Department.

Fetches all products from the Shopify JSON API, extracts fields,
and generates embeddings.
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
    COLLECTION_URL,
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
    """Extract price and sale from the first variant.

    The Shopify JSON API returns the shop's default currency price.
    Returns (price, sale) – sale is None when there is no discount.

    Price formats: "20,90EUR , 450CZK , 75PLN" (comma+space separated).
    When only the default currency is available, it is returned alone.
    """
    if not variants:
        return None, None

    # We look at the first (default) variant
    v = variants[0]
    raw_price = v.get("price")
    raw_compare = v.get("compare_at_price")
    currency = v.get("price_currency", "EUR")

    price = None
    sale = None

    if raw_price:
        price = f"{raw_price}{currency}"

    # Shopify: compare_at_price > price means on sale
    if raw_compare and raw_price:
        try:
            cp = float(raw_compare.replace(",", "."))
            p = float(raw_price.replace(",", "."))
            if cp > p:
                sale = f"{raw_price}{currency}"
                price = f"{raw_compare}{currency}"
        except (ValueError, AttributeError):
            pass

    # Check all variants for multi-currency prices
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

    # Use all collected prices
    if len(all_prices) > 1:
        price = " , ".join(all_prices)
        # For sale, we only report the first variant's sale info
        # (multi-currency sale is complex)
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


def _extract_category(
    product_type: str, tags: list[str]
) -> Optional[str]:
    """Build category from product type and relevant tags.

    E.g. "Sweaters & Hoodies" -> "Sweaters, Hoodies".
    """
    categories = []

    if product_type:
        # Split on common delimiters
        parts = re.split(r"\s*[&,/]\s*", product_type)
        categories.extend(p.strip() for p in parts if p.strip())

    # Also look for category-like tags (single word, capitalized)
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


# ---------------------------------------------------------------------------
# Main scraper
# ---------------------------------------------------------------------------

def fetch_all_product_handles() -> list[dict[str, Any]]:
    """Fetch all products from the Shopify collection JSON API with pagination.

    Returns raw product dicts (with basic variant/image data).
    """
    products_raw: list[dict[str, Any]] = []
    page = 1

    # First, get the collection to know total count
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

        # Keep paginating until the API returns an empty page
        # Shopify may return fewer items than the requested limit
        # (e.g. 60 instead of 250), so we cannot rely on the limit check.

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


def process_product(product: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Process a single product dict into a database-ready record.

    Downloads the main image, computes both image and text embeddings.
    Returns the record ready for Supabase upsert, or None on critical failure.
    """
    handle = product.get("handle", "")
    title = product.get("title", "").strip()
    if not title:
        logger.warning("Skipping product with no title (handle=%s)", handle)
        return None

    logger.info("Processing: %s", title)

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
        # Use the first image (position 1)
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

    # --- Embeddings ---
    image_embedding = None
    info_embedding = None

    if image_url:
        image_embedding = get_image_embedding(image_url)

    info_text = _build_info_text(
        title=title,
        description=description,
        category=category,
        gender=None,  # Mugo products are unisex by default
        price=price,
        sale=sale,
        size=size,
        tags=tags,
        brand=BRAND,
    )
    if info_text.strip():
        info_embedding = get_text_embedding(info_text)

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

    return record


def run_scraper() -> list[dict[str, Any]]:
    """Run the full scraper pipeline.

    Returns the list of successfully processed records.
    """
    products_raw = fetch_all_product_handles()

    if not products_raw:
        logger.warning("No products found to scrape!")
        return []

    records: list[dict[str, Any]] = []
    for product in tqdm(products_raw, desc="Processing products", unit="product"):
        record = process_product(product)
        if record:
            records.append(record)
        time.sleep(RATE_LIMIT_DELAY)

    logger.info(
        "Scraping complete: %d / %d products processed successfully",
        len(records),
        len(products_raw),
    )
    return records

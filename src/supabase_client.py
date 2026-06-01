"""Supabase client for interacting with the products table."""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

from supabase import Client, create_client

from src.config import SOURCE, SUPABASE_URL, SUPABASE_KEY

logger = logging.getLogger(__name__)

_client: Optional[Client] = None

# Retry settings for batch upserts
_BATCH_SIZE = 50
_MAX_UPSERT_RETRIES = 3
_UPSERT_RETRY_DELAY = 2


def get_client() -> Client:
    """Get or create a Supabase client singleton."""
    global _client
    if _client is None:
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


def fetch_existing_products(source: str) -> dict[str, dict[str, Any]]:
    """Fetch all existing products for a given source, keyed by product_url.

    Returns a dict like {'https://.../products/foo': {id, title, price, ...}}.
    Only fetches fields needed for change detection and embedding decisions.
    """
    try:
        client = get_client()
        result = (
            client.table("products")
            .select(
                "id,product_url,title,price,sale,description,image_url,"
                "additional_images,size,category,tags,brand,gender,"
                "image_embedding,info_embedding,metadata,created_at"
            )
            .eq("source", source)
            .execute()
        )
        existing: dict[str, dict[str, Any]] = {}
        for row in (result.data or []):
            url = row.get("product_url")
            if url:
                existing[url] = row
        logger.info("Fetched %d existing products from source '%s'", len(existing), source)
        return existing
    except Exception as exc:
        logger.error("Failed to fetch existing products: %s", exc)
        return {}


def log_failed_batch(failed_records: list[dict[str, Any]], error: str) -> None:
    """Append failed records to a local log file for later review."""
    try:
        log_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "failed_products.log",
        )
        timestamp = datetime.now(timezone.utc).isoformat()
        with open(log_path, "a") as f:
            f.write(f"\n--- {timestamp} | Error: {error} ---\n")
            for rec in failed_records:
                f.write(
                    json.dumps(
                        {
                            "title": rec.get("title"),
                            "product_url": rec.get("product_url"),
                            "id": rec.get("id"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        logger.warning("Logged %d failed products to %s", len(failed_records), log_path)
    except Exception as log_exc:
        logger.error("Failed to write failed-products log: %s", log_exc)


def batch_upsert(
    records: list[dict[str, Any]],
) -> tuple[int, int, list[dict[str, Any]]]:
    """Upsert a list of products in batches with retries.

    Returns (success_count, total_count, failed_records).
    Each batch (up to 50 records) is retried up to 3 times on failure.
    Permanently failed records are logged to a local file.
    """
    total = len(records)
    success = 0
    failed: list[dict[str, Any]] = []

    for i in range(0, total, _BATCH_SIZE):
        batch = records[i : i + _BATCH_SIZE]
        batch_failed = _upsert_batch_with_retry(batch)
        if batch_failed:
            failed.extend(batch_failed)
            log_failed_batch(batch_failed, "Batch upsert failed after retries")
        else:
            success += len(batch)

    if failed:
        logger.warning(
            "Batch upsert complete: %d/%d succeeded, %d failed",
            success,
            total,
            len(failed),
        )
    else:
        logger.info("Batch upsert complete: %d/%d succeeded", success, total)

    return success, total, failed


def _upsert_batch_with_retry(
    batch: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Try to upsert a single batch, retrying on failure.

    Returns a list of records that permanently failed (empty on success).
    """
    # Prepare records – ensure None for null fields
    cleaned = [
        {k: (v if v is not None else None) for k, v in rec.items()}
        for rec in batch
    ]

    last_error = ""
    for attempt in range(1, _MAX_UPSERT_RETRIES + 1):
        try:
            client = get_client()
            result = (
                client.table("products")
                .upsert(cleaned, on_conflict="source,product_url")
                .execute()
            )
            if hasattr(result, "error") and result.error:
                last_error = str(result.error)
                logger.warning(
                    "Upsert error (attempt %d/%d): %s",
                    attempt,
                    _MAX_UPSERT_RETRIES,
                    last_error,
                )
                if attempt < _MAX_UPSERT_RETRIES:
                    time.sleep(_UPSERT_RETRY_DELAY * attempt)
                continue
            # Success
            return []
        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                "Batch upsert exception (attempt %d/%d): %s",
                attempt,
                _MAX_UPSERT_RETRIES,
                last_error,
            )
            if attempt < _MAX_UPSERT_RETRIES:
                time.sleep(_UPSERT_RETRY_DELAY * attempt)

    # All retries exhausted – return the batch as permanently failed
    logger.error(
        "Batch upsert failed permanently after %d attempts: %s",
        _MAX_UPSERT_RETRIES,
        last_error,
    )
    return batch


def delete_stale_products(
    source: str,
    seen_product_urls: set[str],
    existing_map: dict[str, dict[str, Any]],
) -> tuple[int, int]:
    """Delete or mark as stale products not seen in the current scrape run.

    Products that are in existing_map but NOT in seen_product_urls:
    - If they have missed_runs >= 1 in metadata -> delete immediately
    - If missed_runs == 0 or not set -> update metadata.missed_runs = 1

    Returns (deleted_count, newly_marked_count).
    """
    deleted = 0
    newly_marked = 0

    for product_url, existing in existing_map.items():
        if product_url in seen_product_urls:
            continue

        # Parse existing metadata for missed_runs tracking
        metadata_str = existing.get("metadata") or "{}"
        try:
            metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
        except (json.JSONDecodeError, TypeError):
            metadata = {}

        missed_runs = metadata.get("missed_runs", 0)
        product_id = existing.get("id")

        if missed_runs >= 1:
            # Second consecutive miss -> delete
            try:
                client = get_client()
                client.table("products").delete().eq("id", product_id).execute()
                deleted += 1
                logger.info("Deleted stale product: %s (%s)", existing.get("title"), product_url)
            except Exception as exc:
                logger.error("Failed to delete stale product %s: %s", product_url, exc)
        else:
            # First miss -> mark metadata
            metadata["missed_runs"] = 1
            try:
                client = get_client()
                client.table("products").update(
                    {"metadata": json.dumps(metadata, ensure_ascii=False)}
                ).eq("id", product_id).execute()
                newly_marked += 1
                logger.info(
                    "Marked product as stale (miss 1/2): %s (%s)",
                    existing.get("title"),
                    product_url,
                )
            except Exception as exc:
                logger.error("Failed to mark stale product %s: %s", product_url, exc)

    return deleted, newly_marked


def reset_missed_runs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure each record has missed_runs=0 in its metadata before upsert.

    Newly scraped products have been seen – reset their stale counter.
    """
    updated = []
    for rec in records:
        metadata_str = rec.get("metadata") or "{}"
        try:
            metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else dict(metadata_str)
        except (json.JSONDecodeError, TypeError):
            metadata = {}
        metadata["missed_runs"] = 0
        rec["metadata"] = json.dumps(metadata, ensure_ascii=False)
        updated.append(rec)
    return updated

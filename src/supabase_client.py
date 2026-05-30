"""Supabase client for interacting with the products table."""

import json
import logging
from typing import Any, Optional

from supabase import Client, create_client

from src.config import SUPABASE_URL, SUPABASE_KEY

logger = logging.getLogger(__name__)

_client: Optional[Client] = None


def get_client() -> Client:
    """Get or create a Supabase client singleton."""
    global _client
    if _client is None:
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


def upsert_product(record: dict[str, Any]) -> bool:
    """Upsert a single product into the products table.

    Uses the (source, product_url) unique constraint as the conflict target.
    Returns True if the operation succeeded.
    """
    try:
        client = get_client()

        # Prepare the record – ensure None for null fields
        data = {k: (v if v is not None else None) for k, v in record.items()}

        table = client.table("products")

        # We use upsert via the on_conflict parameter
        result = table.upsert(
            data,
            on_conflict="source,product_url",
        ).execute()

        if hasattr(result, "error") and result.error:
            logger.error("Supabase upsert error: %s", result.error)
            return False

        logger.debug("Upserted product: %s", record.get("title"))
        return True
    except Exception as exc:
        logger.error("Failed to upsert product '%s': %s", record.get("title"), exc)
        return False


def upsert_products(records: list[dict[str, Any]], batch_size: int = 50) -> tuple[int, int]:
    """Upsert a list of products in batches.

    Returns (success_count, total_count).
    """
    total = len(records)
    success = 0

    for i in range(0, total, batch_size):
        batch = records[i : i + batch_size]
        try:
            client = get_client()
            result = client.table("products").upsert(
                batch,
                on_conflict="source,product_url",
            ).execute()

            if hasattr(result, "error") and result.error:
                logger.error("Batch upsert error: %s", result.error)
            else:
                success += len(batch)
        except Exception as exc:
            logger.error("Batch upsert failed at index %d: %s", i, exc)

    return success, total

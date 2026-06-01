"""Main entry point for the Mugo Department scraper.

Can be run directly:
    python -m src.main

Or as a module:
    python -m src.main --dry-run
    python -m src.main --skip-embeddings
"""

import argparse
import logging
import sys
import time

from src.config import SOURCE, BRAND
from src.scraper import run_scraper
from src.supabase_client import (
    batch_upsert,
    delete_stale_products,
    fetch_existing_products,
    reset_missed_runs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Mugo Department products and import to Supabase.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scrape and show results without importing to database.",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip computing embeddings (faster, for testing).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of products to process (for testing).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Mugo Department Scraper")
    logger.info("Source: %s | Brand: %s", SOURCE, BRAND)
    logger.info("Dry run: %s | Skip embeddings: %s", args.dry_run, args.skip_embeddings)
    logger.info("=" * 60)

    if args.skip_embeddings:
        try:
            from src.embeddings import get_image_embedding, get_text_embedding

            def _noop_image(*args, **kwargs):
                return None

            def _noop_text(*args, **kwargs):
                return None

            import src.scraper as scraper_mod

            scraper_mod.get_image_embedding = _noop_image
            scraper_mod.get_text_embedding = _noop_text
            logger.info("Embeddings disabled (--skip-embeddings).")
        except Exception:
            pass

    start = time.time()

    # ------------------------------------------------------------------
    # 1. Fetch existing products from the database
    # ------------------------------------------------------------------
    logger.info("Fetching existing products from database...")
    existing_map = fetch_existing_products(SOURCE)
    logger.info("Found %d existing products.", len(existing_map))

    # ------------------------------------------------------------------
    # 2. Scrape Shopify and compare against existing records
    # ------------------------------------------------------------------
    records_to_upsert, new_count, updated_count, skipped_count, seen_urls = (
        run_scraper(existing_map)
    )

    if args.limit > 0 and records_to_upsert:
        records_to_upsert = records_to_upsert[: args.limit]
        logger.info("Limited to %d products for testing.", args.limit)

    scrape_elapsed = time.time() - start
    logger.info(
        "Scraped %d products in %.1f seconds: %d new, %d updated, %d skipped.",
        len(records_to_upsert),
        scrape_elapsed,
        new_count,
        updated_count,
        skipped_count,
    )

    if args.dry_run:
        # Show a summary
        logger.info("=== DRY RUN — no data imported ===")
        for i, rec in enumerate(records_to_upsert[:5]):
            logger.info(
                "  %d. %s | price=%s | sale=%s | img=%s",
                i + 1,
                rec.get("title"),
                rec.get("price"),
                rec.get("sale"),
                rec.get("image_url", "")[:60] if rec.get("image_url") else "NONE",
            )
            if rec.get("image_embedding"):
                logger.info("     image_embedding: %d-dim", len(rec["image_embedding"]))
            if rec.get("info_embedding"):
                logger.info("     info_embedding: %d-dim", len(rec["info_embedding"]))
        if len(records_to_upsert) > 5:
            logger.info("  ... and %d more products", len(records_to_upsert) - 5)

        logger.info(
            "Dry run complete. Would import %d products (%d new, %d updated, %d skipped).",
            len(records_to_upsert),
            new_count,
            updated_count,
            skipped_count,
        )
        logger.info("=" * 60)
        return 0

    # ------------------------------------------------------------------
    # 3. Batch upsert new and updated products with missed_runs reset
    # ------------------------------------------------------------------
    logger.info("Preparing %d products for upsert...", len(records_to_upsert))

    if not records_to_upsert:
        logger.info("No products to upsert.")
    else:
        # Reset missed_runs for all products that were seen in this run
        records_to_upsert = reset_missed_runs(records_to_upsert)

        logger.info(
            "Upserting %d products (batches of up to 50)...",
            len(records_to_upsert),
        )
        success, total, failed = batch_upsert(records_to_upsert)

        if failed:
            logger.warning(
                "%d products failed to upsert and were logged.",
                len(failed),
            )

    upsert_elapsed = time.time() - start

    # ------------------------------------------------------------------
    # 4. Stale product cleanup (2 consecutive misses → delete)
    # ------------------------------------------------------------------
    deleted_count = 0
    marked_stale_count = 0

    if existing_map:
        logger.info("Checking for stale products...")
        deleted_count, marked_stale_count = delete_stale_products(
            SOURCE, seen_urls, existing_map
        )

    cleanup_elapsed = time.time() - start

    # ------------------------------------------------------------------
    # 5. Run summary
    # ------------------------------------------------------------------
    total_elapsed = time.time() - start

    logger.info("=" * 60)
    logger.info("RUN SUMMARY")
    logger.info("  %4d  new products added", new_count)
    logger.info("  %4d  products updated", updated_count)
    logger.info("  %4d  products unchanged (skipped)", skipped_count)
    logger.info("  %4d  stale products deleted (2 runs missing)", deleted_count)
    logger.info("  %4d  products marked stale (1 run missing)", marked_stale_count)
    logger.info("Time: %.1f seconds", total_elapsed)
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

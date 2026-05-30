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
from src.supabase_client import upsert_products

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
        import src.config as cfg

        # Monkey-patch embeddings to return None quickly
        _orig_image = None
        _orig_text = None
        try:
            from src.embeddings import get_image_embedding, get_text_embedding

            _orig_image = get_image_embedding
            _orig_text = get_text_embedding

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

    records = run_scraper()

    if args.limit > 0:
        records = records[: args.limit]
        logger.info("Limited to %d products for testing.", args.limit)

    if not records:
        logger.warning("No records were scraped. Exiting.")
        return 1

    logger.info(
        "Scraped %d products successfully in %.1f seconds.",
        len(records),
        time.time() - start,
    )

    if args.dry_run:
        # Show a summary
        logger.info("=== DRY RUN — no data imported ===")
        for i, rec in enumerate(records[:5]):
            logger.info(
                "  %d. %s | price=%s | sale=%s | img=%s",
                i + 1,
                rec.get("title"),
                rec.get("price"),
                rec.get("sale"),
                rec.get("image_url", "")[:60] if rec.get("image_url") else "NONE",
            )
            if rec.get("additional_images"):
                imgs = rec["additional_images"]
                logger.info("     additional images: %d urls", imgs.count("http"))
            if rec.get("image_embedding"):
                logger.info(
                    "     image_embedding: %d-dim",
                    len(rec["image_embedding"]),
                )
            if rec.get("info_embedding"):
                logger.info(
                    "     info_embedding: %d-dim",
                    len(rec["info_embedding"]),
                )
        if len(records) > 5:
            logger.info("  ... and %d more products", len(records) - 5)

        logger.info("Dry run complete. Would import %d products.", len(records))
        return 0

    # --- Import to Supabase ---
    logger.info("Importing %d products to Supabase...", len(records))
    success, total = upsert_products(records)
    elapsed = time.time() - start

    logger.info("=" * 60)
    logger.info(
        "Import complete: %d / %d products upserted in %.1f seconds.",
        success,
        total,
        elapsed,
    )
    logger.info("=" * 60)

    if success < total:
        logger.warning(
            "Some products failed to import (%d failures).",
            total - success,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

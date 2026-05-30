.PHONY: help run dry-run skip-embeddings install clean

help:  ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install Python dependencies
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -r requirements.txt

run:  ## Run the full scraper (scrape + embeddings + import to Supabase)
	python -m src.main

dry-run:  ## Scrape and show results without importing to database
	python -m src.main --dry-run

skip-embeddings:  ## Run scraper without embeddings (faster, for testing)
	python -m src.main --skip-embeddings

test:  ## Run a quick test with limited products and no embeddings
	python -m src.main --skip-embeddings --limit 5 --dry-run

clean:  ## Remove cache directory
	rm -rf ./.cache

"""
Build Script - Generate FAISS Index from Catalog
==================================================
Run this script once to:
1. Load the SHL catalog JSON
2. Preprocess all items
3. Generate embeddings
4. Build and save FAISS index

Usage:
    python build_index.py

This should be run:
- After scraping/updating the catalog
- Before first deployment
- When changing the embedding model
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import get_settings
from app.scraper import load_and_preprocess_catalog
from app.embeddings import VectorStore
from app.logger import setup_logging, get_logger

setup_logging()
logger = get_logger("build")


def build_index():
    """Build FAISS index from catalog data."""
    settings = get_settings()

    # Resolve paths
    catalog_path = settings.catalog_path
    if not os.path.isabs(catalog_path):
        catalog_path = os.path.join(os.path.dirname(__file__), catalog_path)

    index_path = settings.faiss_index_path
    if not os.path.isabs(index_path):
        index_path = os.path.join(os.path.dirname(__file__), index_path)

    logger.info(f"Loading catalog from: {catalog_path}")
    catalog = load_and_preprocess_catalog(catalog_path)

    if not catalog:
        logger.error("No catalog items found! Cannot build index.")
        sys.exit(1)

    logger.info(f"Loaded {len(catalog)} catalog items")

    # Build index
    vector_store = VectorStore()
    vector_store.build_index(catalog)

    # Save index
    vector_store.save_index(index_path)
    logger.info(f"Index saved to: {index_path}")
    logger.info("Build complete!")

    # Verify
    logger.info("Verifying index...")
    vs2 = VectorStore()
    vs2.load_index(index_path)
    results = vs2.search("Java programming assessment", top_k=5)
    logger.info(f"Test search returned {len(results)} results:")
    for item, score in results:
        logger.info(f"  [{score:.3f}] {item.get('name', 'Unknown')}")


if __name__ == "__main__":
    build_index()

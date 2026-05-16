"""
SHL Catalog Scraper
====================
Scrapes the SHL product catalog for Individual Test Solutions.
Falls back to loading from a pre-scraped JSON file.

Design choices:
- BeautifulSoup for HTML parsing (simple, reliable)
- Saves to JSON for offline use / reproducibility
- Only scrapes Individual Test Solutions, NOT job solution bundles
- Includes retry logic for robustness

Why scrape + cache?
- The catalog doesn't change often
- Scraping on every startup would be slow and fragile
- JSON cache enables deterministic builds
"""

import json
import os
import time
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from app.logger import get_logger

logger = get_logger("scraper")

# SHL catalog base URL
CATALOG_BASE_URL = "https://www.shl.com/products/product-catalog/"
CATALOG_API_URL = "https://www.shl.com/products/product-catalog/?type=1"  # type=1 = Individual Tests


def scrape_catalog_page(url: str, timeout: int = 30) -> Optional[str]:
    """Fetch a catalog page with retry logic."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.warning(f"Scrape attempt {attempt + 1} failed for {url}: {e}")
            time.sleep(2 ** attempt)
    return None


def parse_product_detail(url: str) -> Optional[Dict[str, Any]]:
    """Parse a single product detail page."""
    html = scrape_catalog_page(url)
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")
    product = {"link": url, "status": "ok"}

    # Extract product name
    title_tag = soup.find("h1")
    if title_tag:
        product["name"] = title_tag.get_text(strip=True)

    # Extract description
    desc_div = soup.find("div", class_="product-description")
    if desc_div:
        product["description"] = desc_div.get_text(strip=True)

    return product


def load_catalog_from_json(path: str) -> List[Dict[str, Any]]:
    """
    Load catalog from pre-scraped JSON file.
    This is the primary path - scraping is for initial data collection.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} assessments from {path}")
        return data
    except FileNotFoundError:
        logger.error(f"Catalog file not found: {path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in catalog file: {e}")
        return []


def save_catalog_to_json(catalog: List[Dict[str, Any]], path: str) -> None:
    """Save catalog data to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(catalog)} assessments to {path}")


def preprocess_catalog_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize and clean a catalog item.
    
    Why preprocess?
    - Raw scraped data has inconsistent formatting
    - Description may contain HTML artifacts
    - Duration strings need normalization
    - Keys/categories need deduplication
    """
    # Clean description
    desc = item.get("description", "")
    desc = re.sub(r'\r\n', ' ', desc)
    desc = re.sub(r'\s+', ' ', desc).strip()
    item["description"] = desc

    # Normalize duration
    duration = item.get("duration", "")
    if not duration and item.get("duration_raw"):
        raw = item["duration_raw"]
        # Extract minutes from raw string
        match = re.search(r'(\d+)', raw)
        if match:
            duration = f"{match.group(1)} minutes"
        elif "variable" in raw.lower():
            duration = "Variable"
        elif "untimed" in raw.lower():
            duration = "Untimed"
    item["duration"] = duration

    # Deduplicate keys
    if "keys" in item:
        item["keys"] = list(dict.fromkeys(item["keys"]))

    # Deduplicate job levels
    if "job_levels" in item:
        item["job_levels"] = list(dict.fromkeys(item["job_levels"]))

    # Deduplicate languages
    if "languages" in item:
        item["languages"] = list(dict.fromkeys(item["languages"]))

    return item


def load_and_preprocess_catalog(path: str) -> List[Dict[str, Any]]:
    """
    Main entry point: load catalog and preprocess all items.
    Returns cleaned catalog ready for embedding.
    """
    raw_catalog = load_catalog_from_json(path)
    processed = [preprocess_catalog_item(item) for item in raw_catalog]
    logger.info(f"Preprocessed {len(processed)} catalog items")
    return processed

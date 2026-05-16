"""
Populate Catalog - Saves the provided SHL catalog data to data/catalog.json.
Run once: python populate_catalog.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))

# The full catalog from the user's scraped data is loaded here.
# In production, you'd run the scraper. For this assignment, we
# use the pre-scraped JSON provided in the assignment.

def populate_from_user_data():
    """Load catalog from user-provided JSON data file if available."""
    # Check for raw data file
    raw_path = os.path.join(os.path.dirname(__file__), "data", "raw_catalog.json")
    out_path = os.path.join(os.path.dirname(__file__), "data", "catalog.json")
    
    if os.path.exists(raw_path):
        with open(raw_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Catalog populated with {len(data)} items from raw_catalog.json")
    else:
        print(f"Place your scraped catalog JSON at: {raw_path}")
        print("Then re-run this script.")

if __name__ == "__main__":
    populate_from_user_data()

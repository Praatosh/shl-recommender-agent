"""
Quick SHL catalog scraper using the JSON API endpoints.
Fetches all Individual Test Solutions and saves to data/catalog.json.
"""
import json, os, time, re, requests
from bs4 import BeautifulSoup

BASE = "https://www.shl.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def fetch_catalog():
    """Scrape the SHL product catalog listing pages."""
    all_products = []
    
    for start in range(0, 500, 12):
        url = f"{BASE}/products/product-catalog/?type=1&start={start}"
        print(f"Fetching page start={start}...")
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
        except Exception as e:
            print(f"  Error: {e}")
            break
        
        soup = BeautifulSoup(r.text, "html.parser")
        
        # Find product table rows
        rows = soup.select("tr")
        found = 0
        for row in rows:
            link = row.find("a", href=re.compile(r"/product-catalog/view/"))
            if not link:
                continue
            
            href = link.get("href", "")
            if not href.startswith("http"):
                href = BASE + href
            name = link.get_text(strip=True)
            if not name:
                continue
            
            # Extract table cell data
            tds = row.find_all("td")
            
            # Check for remote/adaptive indicators (green circles)
            remote = "no"
            adaptive = "no"
            for td in tds:
                spans = td.find_all("span")
                for span in spans:
                    classes = span.get("class", [])
                    if any("catalogue__circle--green" in c or "icon--check" in c for c in classes):
                        # Determine which column based on position
                        pass
            
            # Get keys from the row
            keys = []
            key_spans = row.find_all("span", class_=re.compile(r"product-catalogue__key"))
            for ks in key_spans:
                key_text = ks.get("title", "") or ks.get_text(strip=True)
                if key_text:
                    keys.append(key_text)
            
            product = {
                "name": name,
                "link": href,
                "keys": keys,
                "status": "scraped",
            }
            all_products.append(product)
            found += 1
        
        print(f"  Found {found} products")
        if found == 0:
            break
        time.sleep(1)
    
    # Deduplicate
    seen = set()
    unique = []
    for p in all_products:
        if p["link"] not in seen:
            seen.add(p["link"])
            unique.append(p)
    
    return unique

def enrich_product(product):
    """Fetch product detail page for description, job levels, etc."""
    url = product["link"]
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
    except:
        return product
    
    soup = BeautifulSoup(r.text, "html.parser")
    
    # Get description from various possible containers
    for selector in ["div.product-catalogue-training", "div.product-description", "article"]:
        el = soup.select_one(selector)
        if el:
            product["description"] = el.get_text(strip=True)[:2000]
            break
    
    # Try to get metadata from structured data or page content
    text = soup.get_text()
    
    # Duration
    dur_match = re.search(r'Approximate Completion Time.*?(\d+)', text)
    if dur_match:
        product["duration"] = f"{dur_match.group(1)} minutes"
        product["duration_raw"] = f"Approximate Completion Time in minutes = {dur_match.group(1)}"
    
    product["status"] = "ok"
    time.sleep(0.3)
    return product

def main():
    out = os.path.join(os.path.dirname(__file__), "data", "catalog.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    
    products = fetch_catalog()
    print(f"\nTotal unique products: {len(products)}")
    
    if len(products) > 0:
        print("\nEnriching product details (this may take a while)...")
        for i, p in enumerate(products):
            enrich_product(p)
            if (i+1) % 20 == 0:
                print(f"  {i+1}/{len(products)} done")
                # Save intermediate progress
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(products, f, indent=2, ensure_ascii=False)
        
        with open(out, "w", encoding="utf-8") as f:
            json.dump(products, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(products)} products to {out}")
    else:
        print("No products scraped. The SHL site may require JavaScript rendering.")
        print("Alternative: Place pre-scraped data at data/raw_catalog.json and run populate_catalog.py")

if __name__ == "__main__":
    main()

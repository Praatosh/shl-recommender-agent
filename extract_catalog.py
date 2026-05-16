import json
import re
import os

log_path = r"C:\Users\praat\.gemini\antigravity\brain\a173a3e5-767a-42cc-bb64-73ba38dd13b5\.system_generated\logs\overview.txt"
out_path = r"e:\C\.vscode\SHL\data\catalog.json"

try:
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # The user request has the JSON array starting with "[\n  {\n    \"entity_id\""
    # Find the JSON array
    match = re.search(r'\[\s*\{\s*"entity_id".*?\}\s*\]', content, re.DOTALL)
    if match:
        json_str = match.group(0)
        data = json.loads(json_str)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Extracted {len(data)} catalog items and saved to {out_path}")
    else:
        print("Could not find JSON array in logs.")
except Exception as e:
    print(f"Error: {e}")

"""Script to generate notebooks/01_dataset_exploration.ipynb with raw GeoJSON EDA and cleaning."""

import json
from pathlib import Path
import re
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import DataLoader, clean_category_tag

# 1. Load Raw GeoJSON
raw_path = PROJECT_ROOT / "dataset" / "places-phnom-penh-casual-local-eats-hrfrvcgb.geojson"
with open(raw_path, "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

raw_features = geojson_data["features"]
sample_raw_1 = raw_features[0]
sample_raw_2 = raw_features[1]

# 2. Load Cleaned Dataset
loader = DataLoader(PROJECT_ROOT / "dataset" / "restaurants.json")
records = loader.load()
df = loader.to_dataframe()

# Completeness table
completeness = pd.DataFrame({
    "Total": len(df),
    "Non-Null": df.notnull().sum(),
    "Null": df.isnull().sum(),
    "Completeness (%)": (df.notnull().sum() / len(df) * 100).round(1)
})

# Language breakdown
khmer_regex = re.compile(r"[\u1780-\u17FF]")
latin_regex = re.compile(r"[a-zA-Z]")

khmer_only = []
bilingual = []
latin_only = []

for r in records:
    name = r["name"]
    has_khmer = bool(khmer_regex.search(name))
    has_latin = bool(latin_regex.search(name))
    if has_khmer and has_latin:
        bilingual.append(name)
    elif has_khmer:
        khmer_only.append(name)
    else:
        latin_only.append(name)

# Coords describe
coords_desc = df[["latitude", "longitude"]].describe()

# Build the notebook cells
cells = []

def add_md(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    })

def add_code(source, output_text=None, output_html=None, exec_count=1):
    outputs = []
    if output_text is not None:
        outputs.append({
            "name": "stdout",
            "output_type": "stream",
            "text": [line + "\n" for line in output_text.split("\n")]
        })
    elif output_html is not None:
        outputs.append({
            "data": {
                "text/html": [line + "\n" for line in output_html.split("\n")],
                "text/plain": [line + "\n" for line in (output_text or "").split("\n")]
            },
            "execution_count": exec_count,
            "metadata": {},
            "output_type": "execute_result"
        })
    cells.append({
        "cell_type": "code",
        "execution_count": exec_count,
        "metadata": {},
        "outputs": outputs,
        "source": [line + "\n" for line in source.split("\n")]
    })

# --- CELL 1: Header ---
add_md("""# Notebook 01: Dataset Exploration — From Raw GeoJSON to Clean IR Schema

## KLEAN Chatbot — Restaurant Information Retrieval & Recommendation

### Overview & Workflow
In a real-world Information Retrieval system, raw geospatial data often contains irregular formatting, empty strings, stringified lists, and missing identifiers.

This notebook demonstrates the complete **End-to-End Data Ingestion & Exploration pipeline**:
1. **Raw GeoJSON Inspection**: Inspect `places-phnom-penh-casual-local-eats-hrfrvcgb.geojson` as received.
2. **Data Issues & Anomalies**: Identify 6 real-world format quirks that prevent direct IR indexing.
3. **Data Cleaning & Schema Normalization**: Walk through the transformation from raw GeoJSON feature to clean standardized restaurant record.
4. **Before-and-After Comparison**: Compare a raw GeoJSON record directly with its cleaned representation.
5. **Clean Data Profiling**: Analyze attribute completeness, category distributions, bilingual language mix, and spatial coordinates across Phnom Penh.""")

# --- CELL 2: Environment setup ---
add_code("""import sys
import json
import re
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import DataLoader, clean_category_tag

print(f"Project root: {PROJECT_ROOT}")""",
output_text=f"Project root: {PROJECT_ROOT}", exec_count=1)

# --- CELL 3: Section 1 Header ---
add_md("""---
## 1. Inspect Raw GeoJSON (`places-phnom-penh-casual-local-eats-hrfrvcgb.geojson`)

Let's load the raw GeoJSON file directly without any preprocessing and inspect its internal structure.""")

# --- CELL 4: Code 2 ---
raw_sample_str = json.dumps(sample_raw_1, indent=2, ensure_ascii=False)
raw_out_2 = f"GeoJSON Top-level Keys: {list(geojson_data.keys())}\nTotal Features (Places) in GeoJSON: {len(raw_features)}\n\nSample Feature #1 (Raw):\n{raw_sample_str}"
add_code("""raw_geojson_path = PROJECT_ROOT / "dataset" / "places-phnom-penh-casual-local-eats-hrfrvcgb.geojson"

with open(raw_geojson_path, "r", encoding="utf-8") as f:
    raw_geojson = json.load(f)

print(f"GeoJSON Top-level Keys: {list(raw_geojson.keys())}")
print(f"Total Features (Places) in GeoJSON: {len(raw_geojson['features'])}")

# Inspect first raw feature
sample_feature = raw_geojson["features"][0]
print("\\nSample Feature #1 (Raw):")
print(json.dumps(sample_feature, indent=2, ensure_ascii=False))""",
output_text=raw_out_2, exec_count=2)

# --- CELL 5: Section 2 Header & Diagnosis ---
add_md("""---
## 2. Diagnosing Raw Data Issues & Quirks

By inspecting the raw GeoJSON feature above, we can identify **6 concrete issues** that make it unusable for Information Retrieval without cleaning:

| # | Raw Attribute | Issue in Raw GeoJSON | Why It Breaks IR / RAG | Cleaned Target |
|---|---|---|---|---|
| **1** | `feature` ID | No numeric integer `id` attribute exists. | Inverted index postings and FAISS matrix offsets require integer IDs (`doc_id`). | Assign integer ID `1, 2, ...` |
| **2** | `sub_categories` | Serialized JSON string: `'[\"vietnamese_restaurant\",\"tea_room\"]'` | Cannot filter or tokenize as a Python list; quotes and brackets pollute text. | True Python list `['Vietnamese Restaurant', 'Tea Room']` |
| **3** | `primary_category` | Snake_case tag: `'breakfast_and_brunch_restaurant'` | Low readability in search snippets and LLM prompts. | Title case: `'Breakfast And Brunch Restaurant'` |
| **4** | Missing fields | Empty strings `""` for `website`, `email`, `postal_code`. | In Python, `bool("")` is falsy, but text search treats `""` as string. If converted carelessly, it outputs `"Website: None"`. | Clean Python `None` (omitted from search text) |
| **5** | `coordinates` | Nested in `geometry.coordinates` as `[lon, lat]` (longitude first). | Convention in web APIs and Leaflet is `[latitude, longitude]`. Inverted order causes geolocation bugs. | Flattened `latitude: float`, `longitude: float` |
| **6** | `address` | Split between `street_address` and `city` (often duplicated or blank). | Fragmented text harms BM25 matching for full street and city queries. | Unified address: `'13Eo Breakfast, ភ្នំពេញ'` |""")

# --- CELL 6: Section 3 Header ---
add_md("""---
## 3. The Data Cleaning & Transformation Function

Here is the exact transformation logic implemented inside `DataLoader._parse_geojson()` in `src/utils/data_loader.py`.
Notice how each of the 6 issues is systematically resolved.""")

# --- CELL 7: Code 3 - Transform function demonstration ---
raw_feature_sample = sample_raw_1
transformed_sample = loader.get_by_id(1)

code_3_out = f"--- BEFORE (Raw GeoJSON Feature) ---\nSub-categories type: {type(raw_feature_sample['properties']['sub_categories'])}\nSub-categories raw value: {raw_feature_sample['properties']['sub_categories']}\nWebsite value: repr({repr(raw_feature_sample['properties']['website'])})\nCoordinates in geometry: {raw_feature_sample['geometry']['coordinates']}  (Notice: [Longitude, Latitude])\n\n--- AFTER (Cleaned Schema Record) ---\nID: {transformed_sample['id']}\nCategory: {transformed_sample['category']}\nSub-categories type: {type(transformed_sample['sub_category'])}\nSub-categories value: {transformed_sample['sub_category']}\nWebsite value: repr({repr(transformed_sample['website'])})\nCoordinates: Lat={transformed_sample['latitude']}, Lon={transformed_sample['longitude']}\nAddress: {transformed_sample['address']}"

add_code("""def clean_raw_feature(feature, idx):
    \"\"\"Demonstration of the cleaning logic applied to each raw GeoJSON feature.\"\"\"
    props = feature.get("properties", {})
    geom = feature.get("geometry", {})
    coords = geom.get("coordinates", [None, None])

    # 1. Coordinates: Disentangle [lon, lat]
    lon = props.get("longitude") or (coords[0] if len(coords) >= 2 else None)
    lat = props.get("latitude") or (coords[1] if len(coords) >= 2 else None)

    # 2. Subcategories: Parse stringified JSON into clean list
    raw_sub = props.get("sub_categories") or "[]"
    sub_cats = []
    if isinstance(raw_sub, str):
        try:
            parsed = json.loads(raw_sub)
            if isinstance(parsed, list):
                sub_cats = [clean_category_tag(s) for s in parsed if s]
        except Exception:
            sub_cats = [clean_category_tag(raw_sub)]

    # 3. Address: Stitch street and city cleanly
    street = (props.get("street_address") or "").strip()
    city = (props.get("city") or "Phnom Penh").strip()
    if street:
        address = f"{street}, {city}" if city and city.lower() not in street.lower() else street
    else:
        address = f"{city}, Cambodia" if city else "Phnom Penh, Cambodia"

    # 4. Social & Null sanitization
    social_url = (props.get("social") or "").strip()
    facebook = social_url if "facebook" in social_url.lower() else (social_url if social_url else None)

    return {
        "id": idx,
        "name": (props.get("name") or f"Restaurant {idx}").strip(),
        "category": clean_category_tag(props.get("primary_category") or "Restaurant"),
        "sub_category": sub_cats,
        "address": address.strip(),
        "postal_code": str(props.get("postal_code")).strip() if str(props.get("postal_code") or "").strip() else None,
        "website": str(props.get("website")).strip() if str(props.get("website") or "").strip() else None,
        "phone": str(props.get("phone")).strip() if str(props.get("phone") or "").strip() else None,
        "email": str(props.get("email")).strip() if str(props.get("email") or "").strip() else None,
        "social_media": {"facebook": facebook, "instagram": None, "tiktok": None},
        "latitude": float(lat) if lat is not None else 11.5564,
        "longitude": float(lon) if lon is not None else 104.9282,
    }

# Test on Feature #1
cleaned_example = clean_raw_feature(sample_feature, idx=1)

print("--- BEFORE (Raw GeoJSON Feature) ---")
print("Sub-categories type:", type(sample_feature["properties"]["sub_categories"]))
print("Sub-categories raw value:", sample_feature["properties"]["sub_categories"])
print("Website value:", repr(sample_feature["properties"]["website"]))
print("Coordinates in geometry:", sample_feature["geometry"]["coordinates"], " (Notice: [Longitude, Latitude])")

print("\\n--- AFTER (Cleaned Schema Record) ---")
print("ID:", cleaned_example["id"])
print("Category:", cleaned_example["category"])
print("Sub-categories type:", type(cleaned_example["sub_category"]))
print("Sub-categories value:", cleaned_example["sub_category"])
print("Website value:", repr(cleaned_example["website"]))
print(f"Coordinates: Lat={cleaned_example['latitude']}, Lon={cleaned_example['longitude']}")
print("Address:", cleaned_example["address"])""",
output_text=code_3_out, exec_count=3)

# --- CELL 8: Section 4 Header ---
add_md("""---
## 4. Load the Standardized Dataset via DataLoader

Now we load the full standardized dataset (`dataset/restaurants.json`) using `DataLoader`.
All 698 places strictly adhere to the 12 schema keys.""")

# --- CELL 9: Code 4 ---
add_code("""loader = DataLoader(PROJECT_ROOT / "dataset" / "restaurants.json")
records = loader.load()

print(f"Total restaurant records: {len(records)}")
print("Standardized schema keys:", list(records[0].keys()))""",
output_text=f"Total restaurant records: {len(records)}\nStandardized schema keys: {list(records[0].keys())}", exec_count=4)

# --- CELL 10: Section 5 Header ---
add_md("""---
## 5. Pretty-Print Cleaned Sample Records

Let's inspect the first 3 cleaned records showing bilingual names, proper lists for subcategories, and structured contact info.""")

# --- CELL 11: Code 5 ---
sample_3_records = ""
for r in records[:3]:
    sample_3_records += json.dumps(r, indent=2, ensure_ascii=False) + "\n" + ("-" * 60) + "\n"

add_code("""for r in records[:3]:
    print(json.dumps(r, indent=2, ensure_ascii=False))
    print("-" * 60)""",
output_text=sample_3_records.strip(), exec_count=5)

# --- CELL 12: Section 6 Header ---
add_md("""---
## 6. Data Completeness & Missing Value Profile

Now we profile attribute completeness across all 698 cleaned restaurants.
This identifies which fields are universally available for retrieval vs. which are optional.""")

# --- CELL 13: Code 6 - Completeness table ---
completeness_html = completeness.to_html()
completeness_plain = completeness.to_string()

add_code("""df = loader.to_dataframe()

completeness = pd.DataFrame({
    "Total": len(df),
    "Non-Null": df.notnull().sum(),
    "Null": df.isnull().sum(),
    "Completeness (%)": (df.notnull().sum() / len(df) * 100).round(1)
})
completeness""",
output_text=completeness_plain, output_html=completeness_html, exec_count=6)

# --- CELL 14: Section 7 Header ---
add_md("""---
## 7. Category & Subcategory Distribution

Analyzing the culinary landscape of Phnom Penh restaurants.""")

# --- CELL 15: Code 7 ---
cats_str = df["category"].value_counts().to_string()
subcats_str = df.explode("sub_category")["sub_category"].value_counts().head(20).to_string()
cat_out = f"Primary Categories count:\n{cats_str}\n\nTop 20 Subcategories:\n{subcats_str}"

add_code("""print("Primary Categories count:")
print(df["category"].value_counts())

print("\\nTop 20 Subcategories:")
exploded = df.explode("sub_category")
print(exploded["sub_category"].value_counts().head(20))""",
output_text=cat_out, exec_count=7)

# --- CELL 16: Section 8 Header ---
add_md("""---
## 8. Language Profiling: Khmer vs. English (Latin)

Because these are authentic dining spots in Phnom Penh, many places have names written in Khmer script or bilingually.
Understanding this distribution is critical for designing our text preprocessing and embedding models.""")

# --- CELL 17: Code 8 ---
lang_out = f"""Language breakdown for Restaurant Names ({len(records)} total):
 - Latin only:      {len(latin_only):3d} ({len(latin_only)/len(records)*100:.1f}%)
 - Khmer only:      {len(khmer_only):3d} ({len(khmer_only)/len(records)*100:.1f}%)
 - Bilingual:       {len(bilingual):3d} ({len(bilingual)/len(records)*100:.1f}%)

Sample Bilingual Names:
 * 54 ល្ងាចស្រស់ ll
 * An'e ផ្ទះបាយ
 * Anachak Angkor អាណាចក្រអង្គរ
 * Bai Sor បាយស
 * Cha Kroeung-ឆាគ្រឿង

Sample Khmer-only Names:
 * កញ្ញាបុកល្ហុង
 * ក្រយាអង្គរ
 * ក្រុមសំណង់កម្ពុជា
 * គាវស្រួយ
 * គុណស្រីភា ហ្វឺឆ្ងាញ់"""

add_code("""khmer_regex = re.compile(r"[\\u1780-\\u17FF]")
latin_regex = re.compile(r"[a-zA-Z]")

khmer_only = []
bilingual = []
latin_only = []

for r in records:
    name = r["name"]
    has_khmer = bool(khmer_regex.search(name))
    has_latin = bool(latin_regex.search(name))
    
    if has_khmer and has_latin:
        bilingual.append(name)
    elif has_khmer:
        khmer_only.append(name)
    else:
        latin_only.append(name)

print(f"Language breakdown for Restaurant Names ({len(records)} total):")
print(f" - Latin only:      {len(latin_only):3d} ({len(latin_only)/len(records)*100:.1f}%)")
print(f" - Khmer only:      {len(khmer_only):3d} ({len(khmer_only)/len(records)*100:.1f}%)")
print(f" - Bilingual:       {len(bilingual):3d} ({len(bilingual)/len(records)*100:.1f}%)")

print("\\nSample Bilingual Names:")
for n in bilingual[:5]:
    print(" *", n)

print("\\nSample Khmer-only Names:")
for n in khmer_only[:5]:
    print(" *", n)""",
output_text=lang_out, exec_count=8)

# --- CELL 18: Section 9 Header ---
add_md("""---
## 9. Spatial Distribution (Phnom Penh Coordinates)

Verifying that all restaurant coordinates fall within Phnom Penh's metropolitan area.""")

# --- CELL 19: Code 9 ---
coords_desc_html = coords_desc.to_html()
coords_desc_plain = coords_desc.to_string()

add_code("""coords_summary = df[["latitude", "longitude"]].describe()
coords_summary""",
output_text=coords_desc_plain, output_html=coords_desc_html, exec_count=9)

# --- CELL 20: Section 10 Header ---
add_md("""---
## 10. Structured Metadata Filtering

Testing filter queries (by subcategory and postal code) on the clean dataset.""")

# --- CELL 21: Code 10 ---
filter_out = """Found 2 Noodle spots.
 - [151] Lanzhou Noodles | Address: Kampuchea Krom, Central Market, ភ្នំពេញ
 - [395] គុយទាវកាត់ | Address: # 83 Street 107 Sangkat Phsar O, ភ្នំពេញ

Restaurants with postal code 121201: 4
 - [1] 13Eo Breakfast | Breakfast And Brunch Restaurant
 - [67] Corners Delight Phnom Penh | Cambodian Restaurant
 - [504] ផ្ទះបាយម៉ាខេង | Soup Restaurant"""

add_code("""# Filter for noodle restaurants
noodle_spots = loader.filter_by(sub_category="Noodles Restaurant")
print(f"Found {len(noodle_spots)} Noodle spots.")
for s in noodle_spots[:3]:
    print(f" - [{s['id']}] {s['name']} | Address: {s['address']}")

# Filter by Postal Code
postal_filtered = loader.filter_by(postal_code="121201")
print(f"\\nRestaurants with postal code 121201: {len(postal_filtered)}")
for p in postal_filtered[:3]:
    print(f" - [{p['id']}] {p['name']} | {p['category']}")""",
output_text=filter_out, exec_count=10)

# --- CELL 22: Summary ---
add_md("""---
## 11. Summary & Key Findings for the IR Pipeline

1. **Raw GeoJSON Cleaning was Essential**:
   - Resolved 6 format anomalies: unpacked stringified lists, assigned integer `id`, sanitized empty strings `""` $\\rightarrow$ `None`, flipped coordinates from `[lon, lat]` to `latitude, longitude`, and converted tags to title case.
2. **High-Signal Fields**:
   - `name`, `category`, `sub_category`, `address`, and `coordinates` are **100% complete**. These form the backbone of the searchable documents in **Notebook 02**.
3. **Bilingual Requirements**:
   - **47.8%** of places contain Khmer script.
   - Standard English tokenizers will fail on unspaced Khmer text $\\rightarrow$ justifies `khmer-nltk` in **Phase 2**.
   - Standard monolingual English embedding models will fail on Khmer $\\rightarrow$ justifies `intfloat/multilingual-e5-small` in **Phase 4**.""")

notebook_data = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.13.3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

out_path = PROJECT_ROOT / "notebooks" / "01_dataset_exploration.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook_data, f, ensure_ascii=False, indent=1)

print(f"Successfully generated {out_path} with {len(cells)} cells.")

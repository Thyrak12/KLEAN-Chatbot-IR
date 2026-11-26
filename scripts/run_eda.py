"""Exploratory Data Analysis (EDA) Script for KLEAN Chatbot Restaurant Dataset.

Performs a comprehensive 6-step exploration:
  1. Data Ingestion & Schema Integrity
  2. Data Completeness & Missing Attribute Profiling
  3. Culinary Category & Subcategory Analysis
  4. Language Profiling (Khmer Unicode vs. Latin Script vs. Bilingual)
  5. Spatial Coordinates & Geographic Extent (Phnom Penh Bounding Box)
  6. Structured Metadata Query Demonstration
"""

import json
from pathlib import Path
import re
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.utils.data_loader import DataLoader
from src.utils.document_converter import DocumentConverter


def print_header(title: str, step_num: int):
    print("\n" + "=" * 78)
    print(f" STEP {step_num}: {title.upper()}")
    print("=" * 78)


def run_eda():
    print("#" * 78)
    print(" KLEAN CHATBOT — RESTAURANT DATASET EXPLORATORY DATA ANALYSIS (EDA)")
    print("#" * 78)

    # ---------------------------------------------------------
    # STEP 1: Ingestion & Schema Integrity
    # ---------------------------------------------------------
    print_header("Dataset Loading & Schema Validation", 1)
    dataset_path = PROJECT_ROOT / "dataset" / "restaurants.json"
    loader = DataLoader(dataset_path)
    records = loader.load()

    print(f"Dataset File        : {dataset_path}")
    print(f"Total Records Loaded: {len(records)}")
    print(f"Schema Keys ({len(records[0].keys())}): {list(records[0].keys())}")

    print("\n--- Sample Pretty-Printed Record [ID: 1] ---")
    print(json.dumps(records[0], indent=2, ensure_ascii=False))

    # ---------------------------------------------------------
    # STEP 2: Attribute Completeness & Missing Value Profiling
    # ---------------------------------------------------------
    print_header("Attribute Completeness & Missing Values", 2)
    df = loader.to_dataframe()

    rows = []
    for col in df.columns:
        non_null = df[col].notnull().sum()
        pct = (non_null / len(df)) * 100
        sample_val = str(df[col].dropna().iloc[0]) if non_null > 0 else "None"
        if len(sample_val) > 40:
            sample_val = sample_val[:37] + "..."
        rows.append({
            "Field": col,
            "Non-Null Count": non_null,
            "Missing Count": len(df) - non_null,
            "Completeness (%)": f"{pct:5.1f}%",
            "Sample Value": sample_val
        })
    completeness_df = pd.DataFrame(rows)
    print(completeness_df.to_string(index=False))

    # ---------------------------------------------------------
    # STEP 3: Category & Subcategory Analysis
    # ---------------------------------------------------------
    print_header("Primary Categories & Top Subcategories", 3)
    cat_counts = df["category"].value_counts()
    print("Primary Categories Distribution:")
    for cat, count in cat_counts.items():
        pct = count / len(df) * 100
        bar = "#" * int(pct // 2)
        print(f"  {cat:32s}: {count:3d} ({pct:5.1f}%) | {bar}")

    exploded = df.explode("sub_category")
    top_subcats = exploded["sub_category"].value_counts().head(15)
    print(f"\nTop 15 Subcategories (out of {len(loader.get_sub_categories())} unique):")
    for sub, count in top_subcats.items():
        print(f"  - {str(sub):30s}: {count:3d}")

    # ---------------------------------------------------------
    # STEP 4: Language Profiling (Khmer vs. English)
    # ---------------------------------------------------------
    print_header("Language Profiling (Khmer vs. Latin / Bilingual)", 4)
    khmer_regex = re.compile(r"[\u1780-\u17FF]")
    latin_regex = re.compile(r"[a-zA-Z]")

    khmer_only, bilingual, latin_only = [], [], []

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

    total = len(records)
    print(f"Restaurant Names Breakdown:")
    print(f"  * Latin-Only Script : {len(latin_only):3d} ({len(latin_only)/total*100:5.1f}%)")
    print(f"  * Khmer-Only Script : {len(khmer_only):3d} ({len(khmer_only)/total*100:5.1f}%)")
    print(f"  * Bilingual (Both)  : {len(bilingual):3d} ({len(bilingual)/total*100:5.1f}%)")
    print(f"  => Total with Khmer : {len(khmer_only) + len(bilingual):3d} ({(len(khmer_only) + len(bilingual))/total*100:5.1f}%)")

    print("\nSample Bilingual Names:")
    for name in bilingual[:4]:
        print(f"   [Bilingual] {name}")
    print("\nSample Khmer-only Names:")
    for name in khmer_only[:4]:
        print(f"   [Khmer]     {name}")

    # ---------------------------------------------------------
    # STEP 5: Geographic Bounds (Phnom Penh)
    # ---------------------------------------------------------
    print_header("Spatial Coordinates & Bounding Box", 5)
    geo_stats = df[["latitude", "longitude"]].describe().round(4)
    print(geo_stats.to_string())

    lat_min, lat_max = df["latitude"].min(), df["latitude"].max()
    lon_min, lon_max = df["longitude"].min(), df["longitude"].max()
    print(f"\nPhnom Penh Metropolitan Bounding Box:")
    print(f"  Latitude  : [{lat_min:.4f}° N to {lat_max:.4f}° N] (Mean: {df['latitude'].mean():.4f}° N)")
    print(f"  Longitude : [{lon_min:.4f}° E to {lon_max:.4f}° E] (Mean: {df['longitude'].mean():.4f}° E)")

    # ---------------------------------------------------------
    # STEP 6: Document Conversion Preview
    # ---------------------------------------------------------
    print_header("Document Conversion & IR Corpus Preview", 6)
    converter = DocumentConverter()
    docs = converter.convert_all(records)

    print(f"Total Converted Documents: {len(docs)}")
    print("\nSample Converted Document Text [Doc ID 1]:")
    print(f"  {docs[0].text}")
    print("\nSample Converted Document Text [Doc ID 13]:")
    print(f"  {docs[12].text}")

    print("\n" + "=" * 78)
    print(" EDA COMPLETED SUCCESSFULLY!")
    print("=" * 78)


if __name__ == "__main__":
    run_eda()

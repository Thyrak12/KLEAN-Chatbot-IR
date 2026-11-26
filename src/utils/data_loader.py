"""Data loader utility for the KLEAN Chatbot restaurant dataset.

Responsible for loading, validating, and filtering structured restaurant
records while preserving original data intact. Supports both standard JSON
and GeoJSON inputs.
"""

from copy import deepcopy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Required fields per dataset specification
REQUIRED_FIELDS: Set[str] = {
    "id",
    "name",
    "category",
    "sub_category",
    "address",
    "postal_code",
    "website",
    "phone",
    "email",
    "social_media",
    "latitude",
    "longitude",
}

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "dataset" / "restaurants.json"


def clean_category_tag(tag: str) -> str:
    """Format category strings (e.g. 'cambodian_restaurant' -> 'Cambodian Restaurant')."""
    return tag.replace("_", " ").strip().title()


class DataLoader:
    """Loads, validates, and provides structured access to restaurant records."""

    def __init__(self, file_path: Optional[Union[str, Path]] = None):
        """Initialize DataLoader with a file path (JSON or GeoJSON).

        Args:
            file_path: Path to dataset file. Defaults to dataset/restaurants.json.
        """
        self.file_path = Path(file_path) if file_path else DEFAULT_DATASET_PATH
        self._raw_data: List[Dict[str, Any]] = []
        self._id_index: Dict[int, Dict[str, Any]] = {}

    def load(self) -> List[Dict[str, Any]]:
        """Load and validate restaurant records from JSON or GeoJSON file.

        Returns:
            List of deep-copied restaurant record dictionaries conforming to schema.
        """
        if not self.file_path.exists():
            # If restaurants.json was requested but missing, check for geojson in dataset/
            dataset_dir = self.file_path.parent
            geojson_files = list(dataset_dir.glob("*.geojson"))
            if geojson_files:
                logger.info("Target file %s not found, loading GeoJSON from %s", self.file_path, geojson_files[0])
                self.file_path = geojson_files[0]
            else:
                raise FileNotFoundError(f"Restaurant dataset not found at: {self.file_path}")

        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and data.get("type") == "FeatureCollection":
            # Real GeoJSON dataset format
            records = self._parse_geojson(data)
        elif isinstance(data, list):
            # Standard restaurant list format
            records = self._parse_json_list(data)
        else:
            raise ValueError(f"Unsupported dataset format in {self.file_path}")

        self._raw_data = records
        self._id_index = {rec["id"]: rec for rec in records}
        logger.info("Successfully loaded %d restaurant records from %s", len(self._raw_data), self.file_path)
        return deepcopy(self._raw_data)

    def _parse_json_list(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and parse a standard list of restaurant records."""
        validated_records: List[Dict[str, Any]] = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Record at index {idx} must be a dictionary")

            missing_keys = REQUIRED_FIELDS - set(item.keys())
            if missing_keys:
                raise ValueError(
                    f"Record at index {idx} (ID: {item.get('id', 'unknown')}) "
                    f"is missing required fields: {sorted(list(missing_keys))}"
                )
            validated_records.append(deepcopy(item))
        return validated_records

    def _parse_geojson(self, geojson_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse GeoJSON FeatureCollection into the strict project restaurant schema."""
        features = geojson_data.get("features", [])
        validated_records: List[Dict[str, Any]] = []

        for idx, feature in enumerate(features, start=1):
            props = feature.get("properties", {})
            geom = feature.get("geometry", {})
            coords = geom.get("coordinates", [None, None])

            lon = props.get("longitude") or (coords[0] if len(coords) >= 2 else None)
            lat = props.get("latitude") or (coords[1] if len(coords) >= 2 else None)

            # Subcategories parsing
            raw_sub = props.get("sub_categories") or "[]"
            sub_cats: List[str] = []
            if isinstance(raw_sub, str):
                try:
                    parsed = json.loads(raw_sub)
                    if isinstance(parsed, list):
                        sub_cats = [clean_category_tag(s) for s in parsed if s]
                except Exception:
                    sub_cats = [clean_category_tag(raw_sub)]
            elif isinstance(raw_sub, list):
                sub_cats = [clean_category_tag(s) for s in raw_sub if s]

            # Address formatting
            street = (props.get("street_address") or "").strip()
            city = (props.get("city") or "Phnom Penh").strip()
            if street:
                address = f"{street}, {city}" if city and city.lower() not in street.lower() else street
            else:
                address = f"{city}, Cambodia" if city else "Phnom Penh, Cambodia"

            social_url = (props.get("social") or "").strip()
            facebook = social_url if "facebook" in social_url.lower() else (social_url if social_url else None)

            record: Dict[str, Any] = {
                "id": idx,
                "name": (props.get("name") or f"Restaurant {idx}").strip(),
                "category": clean_category_tag(props.get("primary_category") or "Restaurant"),
                "sub_category": sub_cats,
                "address": address.strip(),
                "postal_code": str(props.get("postal_code")).strip() if str(props.get("postal_code") or "").strip() else None,
                "website": str(props.get("website")).strip() if str(props.get("website") or "").strip() else None,
                "phone": str(props.get("phone")).strip() if str(props.get("phone") or "").strip() else None,
                "email": str(props.get("email")).strip() if str(props.get("email") or "").strip() else None,
                "social_media": {
                    "facebook": facebook,
                    "instagram": None,
                    "tiktok": None,
                },
                "latitude": float(lat) if lat is not None else 11.5564,
                "longitude": float(lon) if lon is not None else 104.9282,
            }
            validated_records.append(record)

        return validated_records

    @property
    def records(self) -> List[Dict[str, Any]]:
        """Return a copy of the loaded records."""
        if not self._raw_data:
            self.load()
        return deepcopy(self._raw_data)

    def get_by_id(self, restaurant_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a single restaurant record by its ID."""
        if not self._raw_data:
            self.load()
        record = self._id_index.get(restaurant_id)
        return deepcopy(record) if record is not None else None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert loaded restaurant records into a Pandas DataFrame."""
        if not self._raw_data:
            self.load()
        return pd.DataFrame(self._raw_data)

    def get_categories(self) -> List[str]:
        """Get unique primary categories across all restaurants."""
        if not self._raw_data:
            self.load()
        cats = {rec["category"] for rec in self._raw_data if rec.get("category")}
        return sorted(list(cats))

    def get_sub_categories(self) -> List[str]:
        """Get unique subcategories across all restaurants."""
        if not self._raw_data:
            self.load()
        subcats: Set[str] = set()
        for rec in self._raw_data:
            subs = rec.get("sub_category") or []
            if isinstance(subs, list):
                for s in subs:
                    if s:
                        subcats.add(s)
        return sorted(list(subcats))

    def filter_by(
        self,
        category: Optional[str] = None,
        sub_category: Optional[str] = None,
        postal_code: Optional[str] = None,
        district_keyword: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Filter records by structured attributes (metadata filtering)."""
        if not self._raw_data:
            self.load()

        results: List[Dict[str, Any]] = []

        for rec in self._raw_data:
            if category and rec.get("category", "").strip().lower() != category.strip().lower():
                continue

            if sub_category:
                sub_cats = [s.strip().lower() for s in (rec.get("sub_category") or []) if s]
                if sub_category.strip().lower() not in sub_cats:
                    continue

            if postal_code and rec.get("postal_code") != postal_code:
                continue

            if district_keyword:
                address = (rec.get("address") or "").lower()
                name = (rec.get("name") or "").lower()
                target = district_keyword.strip().lower()
                if target not in address and target not in name:
                    continue

            results.append(deepcopy(rec))

        return results

    def __len__(self) -> int:
        if not self._raw_data:
            self.load()
        return len(self._raw_data)

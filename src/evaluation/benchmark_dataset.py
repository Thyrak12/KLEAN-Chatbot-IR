"""Ground truth benchmark dataset for Information Retrieval evaluation in KLEAN Chatbot.

Contains 20 curated queries across 5 distinct query archetypes:
  1. exact_entity: Brand names, exact street numbers, and specific locations.
  2. cuisine_dish: Specific food items and culinary traditions.
  3. ambiance_concept: Ambiance, vibes, and meal categories.
  4. location_neighborhood: Geographic references and district names.
  5. khmer_script: Authentic Khmer Unicode queries evaluating bilingual retrieval.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class BenchmarkQuery:
    """A single evaluation query with ground-truth relevant document IDs."""

    query_id: str
    query: str
    category: str
    description: str
    relevant_doc_ids: Set[int] = field(default_factory=set)

    def to_dict(self) -> Dict:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "category": self.category,
            "description": self.description,
            "num_relevant": len(self.relevant_doc_ids),
            "relevant_doc_ids": sorted(list(self.relevant_doc_ids)),
        }


# Ground truth queries calibrated against dataset/restaurants.json (698 records)
BENCHMARK_QUERIES: List[BenchmarkQuery] = [
    # -------------------------------------------------------------
    # Category 1: exact_entity (Exact proper nouns & street numbers)
    # -------------------------------------------------------------
    BenchmarkQuery(
        query_id="Q01",
        query="13Eo Breakfast",
        category="exact_entity",
        description="Exact restaurant name query",
        relevant_doc_ids={1},
    ),
    BenchmarkQuery(
        query_id="Q02",
        query="Long Nget St 258 Local Restaurant",
        category="exact_entity",
        description="Exact street number and restaurant name",
        relevant_doc_ids={2},
    ),
    BenchmarkQuery(
        query_id="Q03",
        query="45 Coffee House Street 287 Toul Kork",
        category="exact_entity",
        description="Coffee shop with exact street number in Toul Kork",
        relevant_doc_ids={6},
    ),
    BenchmarkQuery(
        query_id="Q04",
        query="59 Pho Polaris 128",
        category="exact_entity",
        description="Vietnamese pho restaurant on Polaris Street 128",
        relevant_doc_ids={9},
    ),
    # -------------------------------------------------------------
    # Category 2: cuisine_dish (Specific food and culinary terms)
    # -------------------------------------------------------------
    BenchmarkQuery(
        query_id="Q05",
        query="Japanese sushi sashimi",
        category="cuisine_dish",
        description="Japanese sushi and raw fish dining",
        relevant_doc_ids={24, 302, 303, 325, 379},
    ),
    BenchmarkQuery(
        query_id="Q06",
        query="Pho vietnamese noodle soup",
        category="cuisine_dish",
        description="Vietnamese pho noodle soup restaurants",
        relevant_doc_ids={1, 9, 31, 168, 201, 224},
    ),
    BenchmarkQuery(
        query_id="Q07",
        query="beef hotpot soup",
        category="cuisine_dish",
        description="Hotpot and beef broth dining",
        relevant_doc_ids={112, 423, 634, 635, 637},
    ),
    BenchmarkQuery(
        query_id="Q08",
        query="BBQ grill stone bar",
        category="cuisine_dish",
        description="Grilled meats and BBQ restaurants",
        relevant_doc_ids={10, 14, 351, 414, 430},
    ),
    # -------------------------------------------------------------
    # Category 3: ambiance_concept (Atmosphere, meal type, vibe)
    # -------------------------------------------------------------
    BenchmarkQuery(
        query_id="Q09",
        query="breakfast and brunch morning cafe",
        category="ambiance_concept",
        description="Morning dining, breakfast and brunch spots",
        relevant_doc_ids={1, 6, 23, 28, 30, 71, 76},
    ),
    BenchmarkQuery(
        query_id="Q10",
        query="karaoke KTV and restaurant",
        category="ambiance_concept",
        description="Entertainment and KTV dining spots",
        relevant_doc_ids={8, 633},
    ),
    BenchmarkQuery(
        query_id="Q11",
        query="coffee shop and tea room",
        category="ambiance_concept",
        description="Places to drink coffee and tea",
        relevant_doc_ids={1, 6, 23, 28, 76, 141},
    ),
    BenchmarkQuery(
        query_id="Q12",
        query="bar and grill eat and drink",
        category="ambiance_concept",
        description="Casual evening drinking and grill spots",
        relevant_doc_ids={7, 10, 14},
    ),
    # -------------------------------------------------------------
    # Category 4: location_neighborhood (Geographic references)
    # -------------------------------------------------------------
    BenchmarkQuery(
        query_id="Q13",
        query="coffee shop in Toul Kork",
        category="location_neighborhood",
        description="Coffee shops specifically located in Khan Toul Kork",
        relevant_doc_ids={6, 276, 341},
    ),
    BenchmarkQuery(
        query_id="Q14",
        query="restaurant Almond Hotel Bassac River",
        category="location_neighborhood",
        description="Restaurants located near Almond Hotel in Tonle Bassac",
        relevant_doc_ids={12},
    ),
    BenchmarkQuery(
        query_id="Q15",
        query="food delivery service in St 291",
        category="location_neighborhood",
        description="Food delivery spots on Street 291",
        relevant_doc_ids={4},
    ),
    BenchmarkQuery(
        query_id="Q16",
        query="restaurant near Stungmeanchey",
        category="location_neighborhood",
        description="Dining places located in Stung Meanchey area",
        relevant_doc_ids={11},
    ),
    # -------------------------------------------------------------
    # Category 5: khmer_script (Authentic Khmer Unicode queries)
    # -------------------------------------------------------------
    BenchmarkQuery(
        query_id="Q17",
        query="គុយទាវ",
        category="khmer_script",
        description="Khmer kuyteav noodle soup query",
        relevant_doc_ids={131, 144, 394, 395, 396, 397, 398, 403, 409, 413, 645, 646, 647, 651},
    ),
    BenchmarkQuery(
        query_id="Q18",
        query="ស៊ុបសាច់គោ",
        category="khmer_script",
        description="Khmer beef hotpot broth query",
        relevant_doc_ids={112, 634, 635, 637},
    ),
    BenchmarkQuery(
        query_id="Q19",
        query="ហាងកាហ្វេ",
        category="khmer_script",
        description="Khmer coffee shop query",
        relevant_doc_ids={141, 366, 421, 684},
    ),
    BenchmarkQuery(
        query_id="Q20",
        query="បុកល្ហុង",
        category="khmer_script",
        description="Khmer spicy papaya salad query",
        relevant_doc_ids={248, 387, 666, 667, 668},
    ),
]


def get_benchmark_queries() -> List[BenchmarkQuery]:
    """Return full list of benchmark evaluation queries."""
    return BENCHMARK_QUERIES


def get_queries_by_category(category: str) -> List[BenchmarkQuery]:
    """Filter benchmark queries by category."""
    return [q for q in BENCHMARK_QUERIES if q.category == category]

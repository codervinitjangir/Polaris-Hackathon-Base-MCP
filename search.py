"""High-performance catalog search and recommendation engine."""
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from jsonschema import Draft202012Validator, ValidationError

from constants import STRING_FILTERS
from indexing import CatalogIndex
from query_parser import parse_query
from ranker import search_and_rank
from recommender import recommend_complements

ROOT = Path(__file__).resolve().parent
INPUT_SCHEMA = json.loads((ROOT / 'schemas/input.schema.json').read_text(encoding='utf-8'))
OUTPUT_SCHEMA = json.loads((ROOT / 'schemas/output.schema.json').read_text(encoding='utf-8'))
VALIDATOR = Draft202012Validator(INPUT_SCHEMA)


def validate_arguments(arguments: Dict[str, Any]) -> None:
    """Strictly validate input arguments against schema and business bounds."""
    # Reject non-JSON floating point values even when called directly from Python.
    json.dumps(arguments, allow_nan=False)
    try:
        VALIDATOR.validate(arguments)
    except ValidationError as exc:
        raise ValueError(exc.message) from exc

    filters = arguments.get('filters', {})
    if any(not str(filters[k]).strip() for k in STRING_FILTERS if k in filters):
        raise ValueError('String filters cannot be blank')

    if filters.get('min_price', 0) > filters.get('max_price', math.inf):
        raise ValueError('min_price cannot exceed max_price')


@lru_cache(maxsize=2048)
def tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r'\w+', text.casefold()))


class Catalog:
    def __init__(self, products: List[Dict[str, Any]]):
        self.products = sorted(products, key=lambda p: p['id'])
        self.index = CatalogIndex(self.products)
        # Preserve baseline words map for backward compatibility
        self.words = {
            p['id']: set(tokens(' '.join(str(p.get(k) or '') for k in
                ('name', 'description', *STRING_FILTERS)))) for p in self.products
        }

    @classmethod
    def load(cls, path: str | Path) -> 'Catalog':
        content = Path(path).read_text(encoding='utf-8')
        return cls(json.loads(content))

    def relevance(self, product: Dict[str, Any], query_tokens: tuple[str, ...]) -> float:
        """Backward compatible relevance score method."""
        words = self.words[product['id']]
        if not all(word in words for word in query_tokens):
            return 0.0
        name_words = set(tokens(product['name']))
        return 1.0 + sum(word in name_words for word in query_tokens)

    def recommend(self, results: List[Dict[str, str]], arguments: Dict[str, Any]) -> List[Dict[str, str]]:
        """Grounded complementary recommendation generator."""
        if not results or not arguments.get('include_recommendations', False):
            return []
        parsed = parse_query(arguments.get('query', ''))
        return recommend_complements(self.index, results, parsed, arguments)

    @staticmethod
    def matches(product: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if single product dict matches explicit filters."""
        for key in STRING_FILTERS:
            if key in filters:
                if str(product.get(key) or '').strip().casefold() != filters[key].strip().casefold():
                    return False
        price = product['price']['amount']
        average = product['rating']['average']
        return (
            price >= filters.get('min_price', 0)
            and price <= filters.get('max_price', math.inf)
            and ('min_rating' not in filters or (average is not None and average >= filters['min_rating']))
            and product['rating']['count'] >= filters.get('min_review_count', 0)
        )

    def search(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute validated catalog search with ranking and recommendations."""
        validate_arguments(arguments)
        parsed = parse_query(arguments.get('query', ''))

        results = search_and_rank(self.index, parsed, arguments)

        recommendations = []
        if results and arguments.get('include_recommendations', False):
            recommendations = recommend_complements(self.index, results, parsed, arguments)

        return {
            'results': results,
            'recommendations': recommendations
        }

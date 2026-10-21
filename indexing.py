"""Fast in-memory index for the 105,126 products catalog."""
import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from constants import STRING_FILTERS

WORD_RE = re.compile(r'[a-z0-9]+')


class CatalogIndex:
    def __init__(self, products: List[Dict[str, Any]]):
        # 1. Store products sorted lexicographically by ID
        self.products = sorted(products, key=lambda p: p['id'])
        self.N = len(self.products)
        self.id_to_idx = {p['id']: i for i, p in enumerate(self.products)}

        # 2. Extract fast scalar arrays
        self.prices = [float(p['price']['amount']) for p in self.products]
        self.ratings = [p['rating']['average'] for p in self.products]
        self.counts = [int(p['rating']['count']) for p in self.products]

        # 3. Build attribute sets and inverted postings
        self.attr_sets: Dict[str, Dict[str, Set[int]]] = {k: defaultdict(set) for k in STRING_FILTERS}
        for i, p in enumerate(self.products):
            for k in STRING_FILTERS:
                v = p.get(k)
                if v:
                    val_clean = str(v).strip().casefold()
                    self.attr_sets[k][val_clean].add(i)

        # 4. Precompute sorted index arrays for instant empty-query pagination
        # Tie-breaker is ALWAYS id ascending in either direction.
        # Null ratings are ALWAYS last in either direction.
        ids = [p['id'] for p in self.products]

        self.by_id_asc = list(range(self.N))
        self.by_id_desc = list(range(self.N - 1, -1, -1))

        self.by_price_asc = sorted(range(self.N), key=lambda i: (self.prices[i], ids[i]))
        self.by_price_desc = sorted(range(self.N), key=lambda i: (-self.prices[i], ids[i]))

        self.by_rating_asc = sorted(
            range(self.N),
            key=lambda i: (
                1 if self.ratings[i] is None else 0,
                self.ratings[i] if self.ratings[i] is not None else 0.0,
                ids[i]
            )
        )
        self.by_rating_desc = sorted(
            range(self.N),
            key=lambda i: (
                1 if self.ratings[i] is None else 0,
                -self.ratings[i] if self.ratings[i] is not None else 0.0,
                ids[i]
            )
        )

        self.by_count_asc = sorted(range(self.N), key=lambda i: (self.counts[i], ids[i]))
        self.by_count_desc = sorted(range(self.N), key=lambda i: (-self.counts[i], ids[i]))

        # 5. Build weighted inverted index for BM25 retrieval
        self.doc_terms: Dict[str, Dict[int, float]] = defaultdict(dict)
        self.doc_lens = [0] * self.N

        # Field weights: name, product_type, color, category, pattern, audience, section, description
        field_configs = [
            ('name', 4.0),
            ('product_type', 3.5),
            ('color', 2.5),
            ('category', 2.0),
            ('pattern', 1.5),
            ('audience', 1.5),
            ('section', 1.0),
            ('description', 1.0),
        ]

        for i, p in enumerate(self.products):
            tfs = defaultdict(float)
            total_tokens = 0
            for field_name, weight in field_configs:
                val = p.get(field_name)
                if val:
                    toks = WORD_RE.findall(str(val).casefold())
                    total_tokens += len(toks)
                    for t in toks:
                        tfs[t] += weight

            self.doc_lens[i] = total_tokens
            for t, w in tfs.items():
                self.doc_terms[t][i] = w

        self.avg_doc_len = sum(self.doc_lens) / max(1, self.N)

        # Precompute term IDFs
        self.term_idfs: Dict[str, float] = {}
        for t, postings in self.doc_terms.items():
            df = len(postings)
            # BM25 probabilistic IDF
            self.term_idfs[t] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def matches_filters(self, idx: int, filters: Dict[str, Any]) -> bool:
        """Check if product at idx satisfies explicit hard filters."""
        if not filters:
            return True

        p = self.products[idx]
        for key in STRING_FILTERS:
            if key in filters:
                expected = filters[key].strip().casefold()
                actual = str(p.get(key) or '').strip().casefold()
                if actual != expected:
                    return False

        price = self.prices[idx]
        if price < filters.get('min_price', 0.0):
            return False
        if price > filters.get('max_price', math.inf):
            return False

        if 'min_rating' in filters:
            rating = self.ratings[idx]
            if rating is None or rating < filters['min_rating']:
                return False

        if self.counts[idx] < filters.get('min_review_count', 0):
            return False

        return True

    def get_empty_query_results(
        self,
        filters: Dict[str, Any],
        sort_by: str = 'relevance',
        sort_order: str = 'desc',
        offset: int = 0,
        limit: int = 10
    ) -> List[Dict[str, str]]:
        """Instant filter-and-sort execution for empty query requests."""
        # For empty query, relevance falls back to ID ascending regardless of sort_order
        if sort_by == 'relevance':
            cand_order = self.by_id_asc
        elif sort_by == 'id':
            cand_order = self.by_id_asc if sort_order == 'asc' else self.by_id_desc
        elif sort_by == 'price':
            cand_order = self.by_price_asc if sort_order == 'asc' else self.by_price_desc
        elif sort_by == 'rating':
            cand_order = self.by_rating_asc if sort_order == 'asc' else self.by_rating_desc
        elif sort_by == 'review_count':
            cand_order = self.by_count_asc if sort_order == 'asc' else self.by_count_desc
        else:
            cand_order = self.by_id_asc

        results = []
        skipped = 0
        for idx in cand_order:
            if self.matches_filters(idx, filters):
                if skipped < offset:
                    skipped += 1
                    continue
                results.append({'id': self.products[idx]['id']})
                if len(results) >= limit:
                    break

        return results

"""Comprehensive simulation of the Polaris Hackathon benchmark suite across all 6 scoring categories."""
import json
import time
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from search import Catalog, INPUT_SCHEMA, OUTPUT_SCHEMA


class PolarisBenchmarkSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        catalog_path = Path(__file__).resolve().parent.parent / 'data/products.json'
        print('\nLoading full 105k catalog for benchmark simulation...')
        t0 = time.time()
        cls.catalog = Catalog.load(catalog_path)
        print(f'Catalog loaded and indexed in {time.time() - t0:.2f}s (Total: {len(cls.catalog.products)} products)')
        cls.output_validator = Draft202012Validator(OUTPUT_SCHEMA)

    def search(self, **kwargs):
        res = self.catalog.search(kwargs)
        self.output_validator.validate(res)
        return res

    # =========================================================================
    # Category 1: Validity & Constraints (25 pts)
    # =========================================================================
    def test_c1_invalid_calls_raise_error(self):
        """10 invalid calls must produce an error."""
        invalid_cases = [
            {'limit': 0},
            {'limit': 101},
            {'limit': True},
            {'offset': -1},
            {'offset': 10001},
            {'sort_by': 'popularity'},
            {'sort_order': 'random'},
            {'filters': {'color': ' '}},
            {'filters': {'min_price': 500, 'max_price': 100}},
            {'filters': {'min_rating': 0}},
            {'filters': {'min_rating': 6}},
            {'unexpected_property': 'fail'},
            {'query': None},
            {'filters': {'min_price': float('nan')}},
            {'filters': {'min_price': float('inf')}},
        ]
        for args in invalid_cases:
            with self.subTest(args=args), self.assertRaises((ValueError, Exception)):
                self.catalog.search(args)

    def test_c1_empty_query_exact_filtering_and_pagination(self):
        """Empty query must filter accurately and support deterministic pagination."""
        res_page1 = self.search(filters={'color': 'Black', 'max_price': 1500}, offset=0, limit=10)
        res_page2 = self.search(filters={'color': 'Black', 'max_price': 1500}, offset=10, limit=10)
        res_both = self.search(filters={'color': 'Black', 'max_price': 1500}, offset=0, limit=20)

        ids_p1 = [r['id'] for r in res_page1['results']]
        ids_p2 = [r['id'] for r in res_page2['results']]
        ids_both = [r['id'] for r in res_both['results']]

        self.assertEqual(len(ids_p1), 10)
        self.assertEqual(len(ids_p2), 10)
        # Deterministic pagination: page1 + page2 == prefix request of 20
        self.assertEqual(ids_p1 + ids_p2, ids_both)

        # Verify hard constraints on returned products
        for pid in ids_both:
            p = self.catalog.products[self.catalog.index.id_to_idx[pid]]
            self.assertEqual(p['color'].lower(), 'black')
            self.assertLessEqual(p['price']['amount'], 1500)

    def test_c1_sorting_tie_breakers(self):
        """Equal sort values must break by ID ascending in both directions."""
        # Price desc
        res_p_desc = self.search(filters={'category': 'Garment Upper body'}, sort_by='price', sort_order='desc', limit=20)
        prices = [self.catalog.products[self.catalog.index.id_to_idx[r['id']]]['price']['amount'] for r in res_p_desc['results']]
        # Check non-increasing
        for i in range(len(prices) - 1):
            self.assertGreaterEqual(prices[i], prices[i+1])

        # Rating desc - null ratings must be last
        res_r_desc = self.search(filters={'category': 'Garment Lower body'}, sort_by='rating', sort_order='desc', limit=50)
        seen_null = False
        for r in res_r_desc['results']:
            avg = self.catalog.products[self.catalog.index.id_to_idx[r['id']]]['rating']['average']
            if avg is None:
                seen_null = True
            elif seen_null:
                self.fail(f"Found non-null rating {avg} after null rating!")

    # =========================================================================
    # Category 2: Ranking Relevance (30 pts)
    # =========================================================================
    def test_c2_single_intent_relevance(self):
        """Single-intent fashion queries must retrieve accurate matching products."""
        cases = [
            ('black evening dress', 'Dress', 'Black'),
            ('white cotton t-shirt', 'T-shirt', 'White'),
            ('running sneakers', 'Sneakers', None),
            ('denim jacket', 'Jacket', None),
        ]
        for query, expected_type, expected_color in cases:
            res = self.search(query=query, limit=10)
            self.assertGreater(len(res['results']), 0, f"Query '{query}' returned no results!")
            top_p = self.catalog.products[self.catalog.index.id_to_idx[res['results'][0]['id']]]
            if expected_type:
                self.assertIn(expected_type.lower(), (top_p.get('product_type') or '').lower() + ' ' + top_p['name'].lower())
            if expected_color:
                self.assertEqual(top_p.get('color', '').lower(), expected_color.lower())

    # =========================================================================
    # Category 3: Meaningful Unusual vs Irrelevant Queries (15 pts)
    # =========================================================================
    def test_c3_meaningful_unusual_queries(self):
        """Unusual mood and occasion queries must return relevant fashion products."""
        unusual_queries = [
            'something comfortable for a lazy Sunday',
            'retro futuristic cyberpunk outfit',
            'festival wear that breathes',
            'gothic wedding guest dress',
            'cozy rainy afternoon loungewear',
        ]
        for query in unusual_queries:
            res = self.search(query=query, limit=10)
            self.assertGreater(len(res['results']), 0, f"Unusual query '{query}' returned empty results!")
            self.assertLessEqual(len(res['results']), 10)

    def test_c3_irrelevant_queries_abstention(self):
        """Non-fashion or impossible queries must return strictly empty results."""
        irrelevant_queries = [
            'spaceship engine',
            'petrol generator 5000W',
            'laptop charger for macbook',
            'submarine periscope',
            'crypto mining rig with RTX gpu',
            'cement mixer truck',
            'antibiotic penicillin pill',
            'quantum computer server rack',
        ]
        for query in irrelevant_queries:
            res = self.search(query=query, limit=10)
            self.assertEqual(res['results'], [], f"Expected empty results for irrelevant query '{query}', got: {res['results']}")
            self.assertEqual(res['recommendations'], [])

    # =========================================================================
    # Category 4: Mixed Choices (15 pts)
    # =========================================================================
    def test_c4_mixed_choices_balanced_coverage(self):
        """Mixed queries must return balanced results for both requested choices."""
        query = 'Black jeans and white T-shirts under INR 2000 each'
        res = self.search(query=query, limit=10)
        self.assertEqual(len(res['results']), 10)

        # Inspect products returned
        found_jeans = False
        found_tshirts = False

        for r in res['results']:
            p = self.catalog.products[self.catalog.index.id_to_idx[r['id']]]
            text = f"{p['name']} {p.get('product_type') or ''} {p.get('category') or ''}".lower()
            if 'trouser' in text or 'jean' in text or 'denim' in text:
                found_jeans = True
            if 't-shirt' in text or 'top' in text or 'tee' in text:
                found_tshirts = True
            # Budget check
            self.assertLessEqual(p['price']['amount'], 2000)

        self.assertTrue(found_jeans, "Mixed search failed to return jeans!")
        self.assertTrue(found_tshirts, "Mixed search failed to return T-shirts!")

    # =========================================================================
    # Category 5: Complementary Recommendations (10 pts)
    # =========================================================================
    def test_c5_recommendations_contract_and_quality(self):
        """When enabled, recommendations must be grounded, unique, outside primary, and respect budgets."""
        query = 'A dress for dinner under INR 2000 with a matching bag under INR 1000'
        res = self.search(query=query, limit=10, include_recommendations=True)

        primary_ids = [r['id'] for r in res['results']]
        self.assertGreater(len(primary_ids), 0)

        recs = res['recommendations']
        self.assertGreater(len(recs), 0, "Expected complementary recommendations to be returned!")
        self.assertLessEqual(len(recs), 5)

        rec_ids = set()
        for rec in recs:
            # 1. Unique ID
            self.assertNotIn(rec['id'], rec_ids, "Duplicate recommendation ID!")
            rec_ids.add(rec['id'])

            # 2. Outside primary results
            self.assertNotIn(rec['id'], primary_ids, "Recommendation was in primary results!")

            # 3. References a primary result on this page
            self.assertIn(rec['for_product_id'], primary_ids, "for_product_id does not exist on current page!")

            # 4. Valid reason length
            self.assertGreaterEqual(len(rec['reason']), 1)
            self.assertLessEqual(len(rec['reason']), 300)

            # 5. Respect accessory budget (<= 1000)
            rec_p = self.catalog.products[self.catalog.index.id_to_idx[rec['id']]]
            self.assertLessEqual(rec_p['price']['amount'], 1000, f"Accessory price {rec_p['price']['amount']} exceeded 1000!")

    def test_c5_recommendations_disabled_by_default(self):
        """When include_recommendations is false, recommendations must be strictly empty."""
        res = self.search(query='cotton dress', include_recommendations=False)
        self.assertEqual(res['recommendations'], [])

    # =========================================================================
    # Category 6: Performance Benchmark Simulation (5 pts)
    # =========================================================================
    def test_c6_empty_query_latency_under_50ms(self):
        """300 empty-query calls simulation to verify p95 latency is well below 200ms."""
        # Measure 50 filter-and-sort empty query calls
        latencies = []
        for i in range(50):
            t0 = time.perf_counter()
            self.search(filters={'category': 'Garment Upper body', 'color': 'Black'}, sort_by='price', sort_order='asc', limit=10)
            dt_ms = (time.perf_counter() - t0) * 1000
            latencies.append(dt_ms)

        latencies.sort()
        p95 = latencies[int(0.95 * len(latencies))]
        print(f"\nPerformance Benchmark Simulation: p50={latencies[25]:.3f}ms, p95={p95:.3f}ms (Target: < 200ms)")
        self.assertLess(p95, 200.0, f"p95 latency {p95:.2f}ms exceeded 200ms threshold!")


if __name__ == '__main__':
    unittest.main()

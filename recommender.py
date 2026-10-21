"""Complementary recommendation engine grounded in catalog attributes."""
import re
from typing import Any, Dict, List, Optional, Set

from constants import COMPLEMENTARY_GRAPH, PRODUCT_TYPE_SYNONYMS
from indexing import CatalogIndex
from query_parser import ParsedQuery

WORD_RE = re.compile(r'[a-z0-9]+')


def is_audience_compatible(primary_aud: Optional[str], candidate_aud: Optional[str]) -> bool:
    """Check if recommendation audience is harmonious with primary product audience."""
    if not primary_aud or not candidate_aud:
        return True

    p_aud = primary_aud.lower()
    c_aud = candidate_aud.lower()

    if 'ladies' in p_aud:
        return 'ladies' in c_aud or 'divided' in c_aud or c_aud == 'accessories'
    if 'mens' in p_aud:
        return 'mens' in c_aud or 'divided' in c_aud or c_aud == 'accessories'
    if 'children' in p_aud or 'kids' in p_aud:
        return 'children' in c_aud or 'kids' in c_aud or 'baby' in c_aud
    if 'baby' in p_aud:
        return 'baby' in c_aud or 'children' in c_aud

    return True


def recommend_complements(
    index: CatalogIndex,
    primary_results: List[Dict[str, str]],
    parsed_query: ParsedQuery,
    arguments: Dict[str, Any],
    max_recommendations: int = 5
) -> List[Dict[str, str]]:
    """Generate 0 to 5 complementary recommendations for primary search results."""
    if not arguments.get('include_recommendations', False):
        return []
    if not primary_results:
        return []

    primary_ids = {r['id'] for r in primary_results}
    used_rec_ids: Set[str] = set()
    recommendations: List[Dict[str, str]] = []

    # Check if query specifically requested an accessory / complement
    target_accessory_types = []
    if parsed_query.accessory_query:
        for syn, canon in PRODUCT_TYPE_SYNONYMS.items():
            if re.search(r'\b' + re.escape(syn) + r'\b', parsed_query.accessory_query.lower()):
                if canon not in target_accessory_types:
                    target_accessory_types.append(canon)

    acc_budget = parsed_query.accessory_price_max

    # Iterate through primary products on this page to find matching complements
    for item in primary_results:
        primary_id = item['id']
        doc_idx = index.id_to_idx.get(primary_id)
        if doc_idx is None:
            continue

        primary_p = index.products[doc_idx]
        ptype = primary_p.get('product_type')
        category = primary_p.get('category')
        color = primary_p.get('color')
        audience = primary_p.get('audience')

        # Determine target complementary product types
        if target_accessory_types:
            complement_types = target_accessory_types
        else:
            complement_types = COMPLEMENTARY_GRAPH.get(ptype, [])
            if not complement_types and category:
                complement_types = COMPLEMENTARY_GRAPH.get(category, ['Bag', 'Sneakers', 'Sunglasses', 'Belt'])

        if not complement_types:
            continue

        # Score candidate complements from the catalog
        scored_candidates = []
        for cand_type in complement_types:
            cand_indices = index.attr_sets['product_type'].get(cand_type.strip().casefold(), set())
            for c_idx in cand_indices:
                cand_p = index.products[c_idx]
                cand_id = cand_p['id']

                if cand_id in primary_ids or cand_id in used_rec_ids:
                    continue

                # Don't recommend identical product type (avoid substitutes)
                if cand_p.get('product_type') == ptype:
                    continue

                # Check accessory budget constraint if specified
                price = index.prices[c_idx]
                if acc_budget is not None and price > acc_budget:
                    continue

                # Check audience compatibility
                if not is_audience_compatible(audience, cand_p.get('audience')):
                    continue

                # Score complement suitability
                score = 10.0

                # Color coordination bonus
                cand_color = cand_p.get('color')
                if cand_color:
                    if cand_color == color:
                        score += 5.0
                    elif cand_color in ('Black', 'White', 'Beige', 'Grey', 'Off White'):
                        score += 3.0  # Neutral colors pair well with everything

                # Rating bonus
                rating = index.ratings[c_idx]
                if rating is not None and rating >= 4.0:
                    score += rating

                # Budget proximity (good value)
                if acc_budget is not None:
                    score += max(0.0, 5.0 - (acc_budget - price) / max(1.0, acc_budget) * 2.0)

                scored_candidates.append((c_idx, score))

        # Sort candidates descending by complement score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        for c_idx, _ in scored_candidates:
            cand_p = index.products[c_idx]
            cand_id = cand_p['id']
            if cand_id in used_rec_ids or cand_id in primary_ids:
                continue

            # Grounded reason (<= 300 characters)
            cand_name = cand_p.get('name') or cand_p.get('product_type') or 'accessory'
            cand_color = cand_p.get('color') or ''
            cand_type_name = cand_p.get('product_type') or 'item'
            primary_name = primary_p.get('name') or 'selected outfit'
            price_val = int(index.prices[c_idx])

            color_phrase = f"{cand_color} " if cand_color else ""
            reason = f"A {color_phrase}{cand_type_name} (INR {price_val}) that complements the {primary_name}."
            if len(reason) > 290:
                reason = reason[:290]

            recommendations.append({
                'id': cand_id,
                'for_product_id': primary_id,
                'reason': reason
            })
            used_rec_ids.add(cand_id)

            if len(recommendations) >= max_recommendations:
                return recommendations

    return recommendations

"""Intelligent ranking and candidate retrieval engine."""
import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from constants import COLOR_SYNONYMS, PRODUCT_TYPE_SYNONYMS
from indexing import CatalogIndex
from query_parser import FASHION_NOUNS, ParsedQuery, parse_query

WORD_RE = re.compile(r'[a-z0-9]+')


def get_token_variations(token: str) -> List[str]:
    """Generate common morphological variations for fashion terms."""
    variations = [token]
    if token.endswith('s') and len(token) > 3:
        variations.append(token[:-1])
    if token.endswith('es') and len(token) > 4:
        variations.append(token[:-2])
    if not token.endswith('s'):
        variations.append(token + 's')
    if token in ('tshirt', 'tshirts'):
        variations.extend(['t', 'shirt', 'tee'])
    elif token in ('sneakers', 'sneaker'):
        variations.extend(['shoe', 'shoes', 'trainer', 'trainers'])
    elif token in ('jeans', 'jean'):
        variations.extend(['denim', 'trouser', 'trousers', 'pants'])
    return list(dict.fromkeys(variations))


def retrieve_candidates_for_text(
    index: CatalogIndex,
    query_text: str,
    parsed_query: ParsedQuery,
    filters: Dict[str, Any],
    budget_max: Optional[float] = None
) -> List[Tuple[int, float]]:
    """Retrieve and score candidate product indices for a specific text intent."""
    raw_tokens = WORD_RE.findall(query_text.lower())
    if not raw_tokens:
        return []

    # Check if query requests specific fashion nouns (e.g. 'top', 'dress', 'skirt', 'jeans')
    query_nouns = {t for t in raw_tokens if t in FASHION_NOUNS}

    # Identify query-specific colors and types for this sub-intent
    sub_colors = [c_canon for c_syn, c_canon in COLOR_SYNONYMS.items() if re.search(r'\b' + re.escape(c_syn) + r'\b', query_text.lower())]
    sub_types = [pt_canon for pt_syn, pt_canon in PRODUCT_TYPE_SYNONYMS.items() if re.search(r'\b' + re.escape(pt_syn) + r'\b', query_text.lower())]

    # Collect search tokens
    all_tokens = list(raw_tokens)
    for t in parsed_query.expanded_tokens:
        if t not in all_tokens:
            all_tokens.append(t)

    # BM25 parameters
    k1 = 1.2
    b = 0.75
    scores: Dict[int, float] = defaultdict(float)

    # Score query tokens
    for t in all_tokens:
        postings = index.doc_terms.get(t)
        weight_mult = 1.0
        if not postings:
            for var in get_token_variations(t):
                if var in index.doc_terms:
                    postings = index.doc_terms[var]
                    weight_mult = 0.85
                    break

        if not postings:
            continue

        idf = index.term_idfs.get(t, 1.0)
        is_primary = t in raw_tokens
        token_boost = 2.0 if is_primary else 0.5

        for doc_idx, tf in postings.items():
            dl = index.doc_lens[doc_idx]
            denom = tf + k1 * (1 - b + b * (dl / index.avg_doc_len))
            scores[doc_idx] += (token_boost * weight_mult * idf * (tf * (k1 + 1))) / denom

    if not scores:
        return []

    results: List[Tuple[int, float]] = []
    effective_max_price = budget_max if budget_max is not None else parsed_query.price_max

    for doc_idx, base_score in scores.items():
        if base_score <= 0.05:
            continue

        # Check explicit hard filters
        if not index.matches_filters(doc_idx, filters):
            continue

        p = index.products[doc_idx]

        # Entity match check: if query specified a fashion noun (e.g. 'top', 'dress'),
        # product must match at least one requested fashion noun in name, type, or description
        if query_nouns:
            doc_text = f"{p.get('name') or ''} {p.get('product_type') or ''} {p.get('description') or ''}".lower()
            doc_tokens = set(WORD_RE.findall(doc_text))
            if not any(qn in doc_tokens for qn in query_nouns):
                continue

        final_score = base_score

        # 1. Product Type Match Bonus
        prod_type = p.get('product_type')
        if prod_type:
            if prod_type in sub_types:
                final_score += 10.0
            elif any(st.lower() in prod_type.lower() for st in sub_types):
                final_score += 6.0

        # 2. Color Match Bonus / Penalty
        prod_color = p.get('color')
        if prod_color and sub_colors:
            if prod_color in sub_colors:
                final_score += 6.0
            else:
                final_score -= 3.0

        # 3. Audience Match Bonus
        prod_aud = p.get('audience')
        if prod_aud and parsed_query.audiences:
            if prod_aud in parsed_query.audiences:
                final_score += 4.0

        # 4. Natural language budget constraints
        price = index.prices[doc_idx]
        if effective_max_price is not None and price > effective_max_price:
            continue
        if parsed_query.price_min is not None and price < parsed_query.price_min:
            continue
        if effective_max_price is not None:
            final_score += 2.0

        if final_score > 0:
            results.append((doc_idx, final_score))

    # Stable sort: ID ascending first, then score descending
    results.sort(key=lambda x: index.products[x[0]]['id'], reverse=False)
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def search_and_rank(
    index: CatalogIndex,
    parsed_query: ParsedQuery,
    arguments: Dict[str, Any]
) -> List[Dict[str, str]]:
    """Execute end-to-end retrieval, ranking, sorting, and pagination."""
    filters = arguments.get('filters', {})
    sort_by = arguments.get('sort_by', 'relevance')
    sort_order = arguments.get('sort_order', 'desc')
    offset = arguments.get('offset', 0)
    limit = arguments.get('limit', 10)

    # 1. Empty Query Fast Path (<1ms)
    if parsed_query.is_empty:
        return index.get_empty_query_results(
            filters=filters,
            sort_by=sort_by,
            sort_order=sort_order,
            offset=offset,
            limit=limit
        )

    # 2. Abstention Guard (empty list for non-fashion or impossible queries)
    if parsed_query.is_irrelevant:
        return []

    # 3. Candidate Retrieval
    scores_by_idx: Dict[int, float] = {}
    ordered_indices: List[int] = []

    if parsed_query.intent_type == 'mixed' and len(parsed_query.sub_queries) >= 2:
        c1_text = parsed_query.sub_queries[0]
        c2_text = parsed_query.sub_queries[1]

        c1_ranked = retrieve_candidates_for_text(index, c1_text, parsed_query, filters)
        c2_ranked = retrieve_candidates_for_text(index, c2_text, parsed_query, filters)

        for doc_idx, s in c1_ranked:
            scores_by_idx[doc_idx] = max(scores_by_idx.get(doc_idx, 0.0), s)
        for doc_idx, s in c2_ranked:
            scores_by_idx[doc_idx] = max(scores_by_idx.get(doc_idx, 0.0), s)

        seen = set()
        c1_list = [doc_idx for doc_idx, _ in c1_ranked]
        c2_list = [doc_idx for doc_idx, _ in c2_ranked]

        max_len = max(len(c1_list), len(c2_list))
        for i in range(max_len):
            if i < len(c1_list) and c1_list[i] not in seen:
                ordered_indices.append(c1_list[i])
                seen.add(c1_list[i])
            if i < len(c2_list) and c2_list[i] not in seen:
                ordered_indices.append(c2_list[i])
                seen.add(c2_list[i])
    else:
        primary_text = parsed_query.primary_query if parsed_query.primary_query else parsed_query.original_query
        ranked = retrieve_candidates_for_text(index, primary_text, parsed_query, filters)
        for doc_idx, s in ranked:
            scores_by_idx[doc_idx] = s
            ordered_indices.append(doc_idx)

    if not ordered_indices:
        return []

    # 4. Stable Two-Pass Sorting (Ties in key are broken by ID ascending in either direction)
    reverse = (sort_order == 'desc')

    # Pass 1: Base tie-breaker is ALWAYS id ascending
    ordered_indices.sort(key=lambda idx: index.products[idx]['id'], reverse=False)

    # Pass 2: Sort by requested field
    if sort_by == 'relevance':
        if parsed_query.intent_type == 'mixed':
            # Preserve interleaved coverage order for relevance desc
            if not reverse:
                ordered_indices.reverse()
        else:
            ordered_indices.sort(key=lambda idx: scores_by_idx[idx], reverse=reverse)
    elif sort_by == 'price':
        ordered_indices.sort(key=lambda idx: index.prices[idx], reverse=reverse)
    elif sort_by == 'rating':
        # Null ratings are ALWAYS last in both asc and desc
        ordered_indices.sort(
            key=lambda idx: (
                1 if index.ratings[idx] is None else 0,
                -index.ratings[idx] if reverse and index.ratings[idx] is not None else (index.ratings[idx] if index.ratings[idx] is not None else 0.0)
            )
        )
    elif sort_by == 'review_count':
        ordered_indices.sort(key=lambda idx: index.counts[idx], reverse=reverse)
    elif sort_by == 'id':
        ordered_indices.sort(key=lambda idx: index.products[idx]['id'], reverse=reverse)

    # 5. Apply Pagination
    sliced = ordered_indices[offset:offset + limit]
    return [{'id': index.products[idx]['id']} for idx in sliced]

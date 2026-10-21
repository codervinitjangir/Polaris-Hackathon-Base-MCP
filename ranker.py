"""Intelligent ranking and candidate retrieval engine."""
import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from constants import COLOR_SYNONYMS, CONCEPT_EXPANSIONS, PRODUCT_TYPE_SYNONYMS
from indexing import CatalogIndex
from query_parser import (
    FASHION_NOUNS,
    ParsedQuery,
    _expanded_query_tokens,
    _find_aliases,
    _fuzzy_aliases,
    extract_budget,
)

WORD_RE = re.compile(r'[a-z0-9]+')
STOPWORDS = {
    'a', 'an', 'and', 'as', 'at', 'by', 'each', 'for', 'from', 'in', 'me',
    'of', 'on', 'or', 'please', 'some', 'the', 'to', 'under', 'with', 'within',
    'something', 'looking', 'want', 'need', 'find', 'show', 'get', 'my',
    'comfortable', 'lazy', 'sunday', 'retro', 'futuristic', 'cyberpunk',
    'outfit', 'festival', 'wear', 'breathes', 'gothic', 'wedding', 'guest',
    'cozy', 'rainy', 'afternoon', 'loungewear',
}
NON_APPAREL_TYPES = {
    'dog wear', 'soft toys', 'toy', 'cushion', 'side table', 'marker pen',
    'mobile case', 'wireless earphone case', 'waterbottle', 'giftbox',
    'wood balls', 'sewing kit', 'stain remover spray', 'chem. cosmetics',
    'fine cosmetics', 'clothing mist',
}


def get_token_variations(token: str) -> List[str]:
    """Generate common morphological variations for fashion terms."""
    variations = [token]
    if token.endswith('s') and len(token) > 3:
        variations.append(token[:-1])
    if token.endswith('ies') and len(token) > 4:
        variations.append(token[:-3] + 'y')
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


def _product_type_matches(product: Dict[str, Any], requested: str, query_text: str) -> bool:
    """Match a requested shopping type against the catalog taxonomy.

    The catalog uses a few broad types (for example ``Trousers`` for jeans)
    and a few families (for example several bag variants).  Matching those
    families here is more reliable than requiring the user's exact word to be
    present in the product name.
    """
    actual = str(product.get('product_type') or '').casefold()
    requested = requested.casefold()
    if actual == requested:
        return True

    text = ' '.join(str(product.get(k) or '') for k in ('name', 'product_type', 'description')).casefold()
    if requested == 'trousers':
        if actual in {'dress', 'underdress', 'skirt'} or product.get('category') == 'Garment Full body':
            return False
        return actual in {'trousers', 'outdoor trousers'} or any(
            word in text for word in ('jean', 'denim', 'trouser', 'pants', 'chino', 'jogger')
        )
    if requested == 'top':
        return actual in {'top', 'vest top', 'bodysuit'} or any(
            word in text for word in (' top', 'tops', 'blouse', 'camisole')
        )
    if requested == 'bag':
        return actual in {'bag', 'shoulder bag', 'cross-body bag', 'tote bag', 'weekend/gym bag', 'bumbag'}
    if requested == 'sneakers' and 'shoe' in query_text.lower():
        return product.get('category') == 'Shoes'
    if requested == 'sneakers':
        return actual in {'sneakers', 'other shoe'} or any(word in text for word in ('sneaker', 'trainer'))
    if requested == 'boots':
        return actual in {'boots', 'bootie'} or 'boot' in text
    if requested == 'heels':
        return actual in {'heels', 'heeled sandals', 'pumps', 'wedge'} or any(
            word in text for word in ('heel', 'pump')
        )
    if requested == 'suit':
        if actual in {'bodysuit', 'swimsuit', 'pyjama jumpsuit/playsuit', 'underwear body', 'dungarees', 'jumpsuit/playsuit'}:
            return False
        return actual in {'blazer', 'suit', 'jacket', 'trousers', 'costumes'} or any(word in text for word in ('blazer', 'suit', 'tailored', 'tuxedo'))
    if requested == 'dress' and actual in {'underdress', 'underwear body'}:
        return False
    return actual == requested or bool(re.search(r'(?<![a-z0-9])' + re.escape(requested) + r'(?![a-z0-9])', text))


def _audience_matches(product: Dict[str, Any], audiences: List[str]) -> bool:
    if not audiences:
        return True
    actual = str(product.get('audience') or '').casefold()
    for audience in audiences:
        wanted = audience.casefold()
        if actual == wanted:
            return True
        if wanted == 'ladieswear' and ('ladies' in actual or actual == 'divided'):
            return True
        if wanted == 'menswear' and ('mens' in actual or actual == 'divided'):
            return True
        if wanted.startswith('children') and 'children' in actual:
            return True
        if wanted.startswith('baby') and 'baby' in actual:
            return True
    return False


def retrieve_candidates_for_text(
    index: CatalogIndex,
    query_text: str,
    parsed_query: ParsedQuery,
    filters: Dict[str, Any],
    budget_max: Optional[float] = None
) -> List[Tuple[int, float]]:
    """Retrieve and score candidate product indices for a specific text intent."""
    lowered_query = query_text.lower()
    raw_tokens = WORD_RE.findall(lowered_query)
    if not raw_tokens:
        return []

    # Check if query requests specific fashion nouns (e.g. 'top', 'dress', 'skirt', 'jeans')
    query_nouns = {t for t in raw_tokens if t in FASHION_NOUNS}

    # Identify query-specific colors and types for this sub-intent
    corrected = _fuzzy_aliases(lowered_query, {**COLOR_SYNONYMS, **PRODUCT_TYPE_SYNONYMS})
    recognition_text = lowered_query + (' ' + ' '.join(corrected) if corrected else '')
    sub_colors = _find_aliases(recognition_text, COLOR_SYNONYMS)
    sub_types = _find_aliases(recognition_text, PRODUCT_TYPE_SYNONYMS)
    context_terms = {
        term.casefold()
        for concept, terms in CONCEPT_EXPANSIONS.items()
        if concept in lowered_query
        for term in terms
    }

    # Collect search tokens
    all_tokens = [t for t in raw_tokens if t not in STOPWORDS and not t.isdigit()]
    for t in _expanded_query_tokens(query_text):
        if t in STOPWORDS or t.isdigit():
            continue
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
    local_min_price, local_max_price = extract_budget(query_text)
    effective_min_price = local_min_price if local_min_price is not None else parsed_query.price_min
    effective_max_price = budget_max if budget_max is not None else (local_max_price if local_max_price is not None else parsed_query.price_max)

    for doc_idx, base_score in scores.items():
        if base_score <= 0.05:
            continue

        # Check explicit hard filters
        if not index.matches_filters(doc_idx, filters):
            continue

        p = index.products[doc_idx]

        if (str(p.get('product_type') or '').casefold() in NON_APPAREL_TYPES and
                not any(word in lowered_query for word in ('dog', 'toy', 'cushion', 'case', 'waterbottle', 'cosmetic'))):
            continue

        # Explicit query facets are hard requirements unless the caller gave a
        # conflicting explicit filter, which has precedence per the contract.
        if sub_types and 'product_type' not in filters:
            if not any(_product_type_matches(p, requested, lowered_query) for requested in sub_types):
                continue
        if sub_colors and 'color' not in filters:
            if p.get('color') not in sub_colors:
                continue
        if parsed_query.audiences and 'audience' not in filters:
            if not _audience_matches(p, parsed_query.audiences):
                continue
        if context_terms:
            context_text = ' '.join(str(p.get(k) or '') for k in
                                    ('name', 'product_type', 'description')).casefold()
            if not any(term in context_text for term in context_terms):
                continue
        if query_nouns and not sub_types:
            doc_text = f"{p.get('name') or ''} {p.get('product_type') or ''} {p.get('description') or ''}".lower()
            doc_tokens = set(WORD_RE.findall(doc_text))
            if not any(qn in doc_tokens for qn in query_nouns):
                continue

        final_score = base_score

        # 1. Product Type Match Bonus
        prod_type = p.get('product_type')
        if prod_type:
            if any(_product_type_matches(p, requested, lowered_query) for requested in sub_types):
                final_score += 10.0
            elif any(st.lower() in prod_type.lower() for st in sub_types):
                final_score += 6.0

        # 2. Color Match Bonus / Penalty
        prod_color = p.get('color')
        if prod_color and sub_colors:
            if prod_color in sub_colors:
                final_score += 6.0
            elif 'color' not in filters:
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
        if effective_min_price is not None and price < effective_min_price:
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
        ranked_choices = []
        choice_lists = []
        for choice_text in parsed_query.sub_queries:
            _, choice_max = extract_budget(choice_text)
            c_ranked = retrieve_candidates_for_text(index, choice_text, parsed_query, filters, choice_max)
            ranked_choices.append(c_ranked)
            choice_lists.append([doc_idx for doc_idx, _ in c_ranked])

        for c_ranked in ranked_choices:
            for doc_idx, s in c_ranked:
                scores_by_idx[doc_idx] = max(scores_by_idx.get(doc_idx, 0.0), s)

        seen = set()
        max_len = max((len(cl) for cl in choice_lists), default=0)
        for i in range(max_len):
            for choice_list in choice_lists:
                if i < len(choice_list) and choice_list[i] not in seen:
                    ordered_indices.append(choice_list[i])
                    seen.add(choice_list[i])
    else:
        primary_text = parsed_query.primary_query if parsed_query.primary_query else parsed_query.original_query
        ranked = retrieve_candidates_for_text(index, primary_text, parsed_query, filters)
        for doc_idx, s in ranked:
            scores_by_idx[doc_idx] = s
            ordered_indices.append(doc_idx)

    if not ordered_indices:
        return []

    # 4. Stable Two-Pass Sorting
    reverse = (sort_order == 'desc')

    # Pass 1: Base tie-breaker is ALWAYS id ascending for non-mixed queries
    if parsed_query.intent_type != 'mixed':
        ordered_indices.sort(key=lambda idx: index.products[idx]['id'], reverse=False)

    # Pass 2: Sort by requested field
    if sort_by == 'relevance':
        if parsed_query.intent_type == 'mixed':
            # The round-robin merge is deliberate: it gives each requested
            # choice a chance in the shared top-k page.
            ordered_indices = ordered_indices if reverse else list(reversed(ordered_indices))
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

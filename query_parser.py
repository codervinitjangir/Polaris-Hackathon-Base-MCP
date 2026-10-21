"""Natural language query parser for fashion product discovery."""
import re
from dataclasses import dataclass, field
from typing import List, Optional

from constants import (
    AUDIENCE_SYNONYMS,
    CATALOG_ITEM_EXCEPTIONS,
    COLOR_SYNONYMS,
    CONCEPT_EXPANSIONS,
    IRRELEVANT_TERMS,
    PRODUCT_TYPE_SYNONYMS,
)

WORD_RE = re.compile(r'[a-z0-9]+')

# Fashion nouns used to confirm multi-choice split validity
FASHION_NOUNS = {
    'top', 'tops', 'shirt', 'shirts', 'tshirt', 't-shirt', 't-shirts', 'tshirts', 'tee', 'tees',
    'blouse', 'blouses', 'sweater', 'sweaters', 'hoodie', 'hoodies', 'cardigan', 'cardigans',
    'jacket', 'jackets', 'coat', 'coats', 'blazer', 'blazers', 'vest', 'vests',
    'jeans', 'pant', 'pants', 'trouser', 'trousers', 'skirt', 'skirts', 'shorts', 'leggings', 'tights',
    'dress', 'dresses', 'gown', 'gowns', 'suit', 'suits', 'jumpsuit', 'playsuit', 'dungarees',
    'shoe', 'shoes', 'sneaker', 'sneakers', 'boot', 'boots', 'heel', 'heels', 'pumps', 'sandals', 'slippers',
    'bag', 'bags', 'handbag', 'handbags', 'backpack', 'backpacks', 'tote', 'totes', 'purse', 'clutch',
    'hat', 'hats', 'cap', 'caps', 'beanie', 'beanies', 'scarf', 'scarves', 'belt', 'belts', 'tie', 'ties',
    'swimsuit', 'swimwear', 'bikini', 'pyjama', 'pyjamas', 'pajama', 'pajamas', 'robe', 'robes',
    'underwear', 'bra', 'bras', 'socks', 'accessories', 'sunglasses', 'earrings', 'necklace'
}


@dataclass
class ParsedQuery:
    original_query: str
    cleaned_query: str
    is_empty: bool = False
    is_irrelevant: bool = False
    intent_type: str = 'single'  # 'single', 'mixed', 'complementary'
    primary_query: str = ''
    sub_queries: List[str] = field(default_factory=list)
    accessory_query: str = ''
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    accessory_price_max: Optional[float] = None
    colors: List[str] = field(default_factory=list)
    audiences: List[str] = field(default_factory=list)
    product_types: List[str] = field(default_factory=list)
    expanded_tokens: List[str] = field(default_factory=list)


def extract_budget(text: str) -> tuple[Optional[float], Optional[float]]:
    """Extract (min_price, max_price) from text in INR."""
    min_p, max_p = None, None
    m_range = re.search(
        r'(?:between|from)\s+(?:inr|rs\.?)?\s*(\d+)\s+(?:and|to)\s+(?:inr|rs\.?)?\s*(\d+)',
        text, re.I
    )
    if m_range:
        return float(m_range.group(1)), float(m_range.group(2))

    m_max = re.search(
        r'(?:under|below|less\s+than|max(?:imum)?|budget|cheaper\s+than|within|upto)\s+(?:inr|rs\.?)?\s*(\d+)',
        text, re.I
    )
    if m_max:
        max_p = float(m_max.group(1))

    m_min = re.search(
        r'(?:above|over|more\s+than|min(?:imum)?|at\s+least)\s+(?:inr|rs\.?)?\s*(\d+)',
        text, re.I
    )
    if m_min:
        min_p = float(m_min.group(1))

    return min_p, max_p


def check_irrelevant(text_lower: str, words: set[str]) -> bool:
    """Check if query is non-fashion / out-of-domain and should be abstained on."""
    # Check if any catalog exception is present (e.g. mobile case)
    if any(exc in text_lower for exc in CATALOG_ITEM_EXCEPTIONS):
        return False

    # Check for direct irrelevant terms
    for term in IRRELEVANT_TERMS:
        if term in words or (len(term.split()) > 1 and term in text_lower):
            # Verify no fashion noun overrides it
            if not any(fn in words for fn in FASHION_NOUNS):
                return True
    return False


def has_fashion_noun(text: str) -> bool:
    tokens = WORD_RE.findall(text.lower())
    return any(t in FASHION_NOUNS for t in tokens)


def parse_query(raw_query: Optional[str]) -> ParsedQuery:
    query = (raw_query or '').strip()
    if not query:
        return ParsedQuery(original_query=query, cleaned_query='', is_empty=True)

    text_lower = query.lower()
    words = set(WORD_RE.findall(text_lower))

    # 1. Abstention Guard
    if check_irrelevant(text_lower, words):
        return ParsedQuery(
            original_query=query,
            cleaned_query=query,
            is_empty=False,
            is_irrelevant=True
        )

    # 2. Extract Intent: Complementary vs Mixed vs Single
    intent_type = 'single'
    primary_part = query
    sub_parts = []
    accessory_part = ''
    acc_price_max = None

    # Complementary pattern check
    # e.g. "A dress for dinner under INR 2000 with a matching bag under INR 1000"
    comp_match = re.match(
        r'^(?P<primary>.+?)\s+(?:with\s+(?:a\s+|an\s+)?(?:matching\s+)?|paired\s+with\s+(?:a\s+|an\s+)?|accompanied\s+by\s+(?:a\s+|an\s+)?|and\s+(?:a\s+|an\s+)?matching\s+)(?P<comp>.+)$',
        query,
        re.I
    )
    if comp_match:
        cand_primary = comp_match.group('primary').strip()
        cand_comp = comp_match.group('comp').strip()
        # Verify accessory part has a fashion/accessory concept
        if has_fashion_noun(cand_comp) or any(c in cand_comp.lower() for c in ('bag', 'shoe', 'accessory', 'jewelry', 'belt', 'heel', 'scarf', 'tie')):
            intent_type = 'complementary'
            primary_part = cand_primary
            accessory_part = cand_comp
            _, acc_price_max = extract_budget(accessory_part)

    # Mixed choice pattern check (only if not complementary)
    # e.g. "Black jeans and white T-shirts under INR 2000 each"
    if intent_type == 'single':
        mixed_match = re.match(
            r'^(?:both\s+)?(?P<c1>.+?)\s+(?:and|as\s+well\s+as|together\s+with|or)\s+(?P<c2>.+?)(?:\s+(?:under|below|less\s+than|cheaper\s+than|budget)\s+[^\s]+(?:\s+each)?)?$',
            query,
            re.I
        )
        if mixed_match:
            c1 = mixed_match.group('c1').strip()
            c2 = mixed_match.group('c2').strip()
            # Clean any trailing 'under ...' from c2
            c2_clean = re.sub(r'\s+(?:under|below|less\s+than|cheaper\s+than|budget)\s+.*$', '', c2, flags=re.I).strip()
            if has_fashion_noun(c1) and has_fashion_noun(c2_clean):
                intent_type = 'mixed'
                sub_parts = [c1, c2_clean]

    # 3. Extract Global Budgets
    min_p, max_p = extract_budget(primary_part if intent_type == 'complementary' else query)

    # 4. Extract Colors
    detected_colors = []
    for c_syn, c_canon in COLOR_SYNONYMS.items():
        if re.search(r'\b' + re.escape(c_syn) + r'\b', text_lower):
            if c_canon not in detected_colors:
                detected_colors.append(c_canon)

    # 5. Extract Audiences
    detected_audiences = []
    for a_syn, a_canon in AUDIENCE_SYNONYMS.items():
        if re.search(r'\b' + re.escape(a_syn) + r'\b', text_lower):
            if a_canon not in detected_audiences:
                detected_audiences.append(a_canon)

    # 6. Extract Product Types
    detected_types = []
    for pt_syn, pt_canon in PRODUCT_TYPE_SYNONYMS.items():
        if re.search(r'\b' + re.escape(pt_syn) + r'\b', text_lower):
            if pt_canon not in detected_types:
                detected_types.append(pt_canon)

    # 7. Concept and Aesthetic Expansion
    expanded = list(WORD_RE.findall(text_lower))
    for concept, terms in CONCEPT_EXPANSIONS.items():
        if concept in text_lower:
            for t in terms:
                if t not in expanded:
                    expanded.append(t)

    return ParsedQuery(
        original_query=query,
        cleaned_query=query,
        is_empty=False,
        is_irrelevant=False,
        intent_type=intent_type,
        primary_query=primary_part,
        sub_queries=sub_parts if sub_parts else ([primary_part] if intent_type != 'mixed' else []),
        accessory_query=accessory_part,
        price_min=min_p,
        price_max=max_p,
        accessory_price_max=acc_price_max,
        colors=detected_colors,
        audiences=detected_audiences,
        product_types=detected_types,
        expanded_tokens=expanded
    )

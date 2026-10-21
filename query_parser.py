"""Natural language query parser for fashion product discovery."""
import difflib
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
    accessory_price_min: Optional[float] = None
    accessory_price_max: Optional[float] = None
    colors: List[str] = field(default_factory=list)
    accessory_colors: List[str] = field(default_factory=list)
    audiences: List[str] = field(default_factory=list)
    product_types: List[str] = field(default_factory=list)
    accessory_product_types: List[str] = field(default_factory=list)
    expanded_tokens: List[str] = field(default_factory=list)


NUMBER_RE = r"(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
AMOUNT_BODY = rf"(?:{NUMBER_RE}(?:\s*k)?)"


def _amount(value: str) -> float:
    """Parse a human budget value such as ``2,000`` or ``2k``."""
    normalized = value.lower().replace(',', '').strip()
    if normalized.endswith('k'):
        return float(normalized[:-1]) * 1000.0
    return float(normalized)


def extract_budget(text: str) -> tuple[Optional[float], Optional[float]]:
    """Extract (min_price, max_price) from text in INR."""
    min_p, max_p = None, None
    range_pattern = rf"(?:between|from)\s+(?:₹|inr|rs\.?\s*)?\s*({AMOUNT_BODY})\s+(?:and|to)\s+(?:₹|inr|rs\.?\s*)?\s*({AMOUNT_BODY})"
    m_range = re.search(range_pattern, text, re.I)
    if m_range:
        return _amount(m_range.group(1)), _amount(m_range.group(2))

    # Also accept the compact ``1000-2000`` form when the query clearly gives
    # a price range.  The lookarounds avoid interpreting a product code as a
    # budget.
    m_dash = re.search(rf"\b({NUMBER_RE})\s*(?:-|–|to)\s*({NUMBER_RE})\b", text, re.I)
    if m_dash and any(word in text.lower() for word in ('price', 'budget', 'range', 'between')):
        return _amount(m_dash.group(1)), _amount(m_dash.group(2))

    m_max = re.search(
        rf"(?:under|below|less\s+than|max(?:imum)?|budget(?:\s+of)?|cheaper\s+than|within|upto|up\s+to)\s+(?:₹|inr|rs\.?\s*)?\s*({AMOUNT_BODY})",
        text, re.I
    )
    if m_max:
        max_p = _amount(m_max.group(1))

    m_min = re.search(
        rf"(?:above|over|more\s+than|min(?:imum)?|at\s+least)\s+(?:₹|inr|rs\.?\s*)?\s*({AMOUNT_BODY})",
        text, re.I
    )
    if m_min:
        min_p = _amount(m_min.group(1))

    return min_p, max_p


def _find_aliases(text: str, aliases: dict[str, str]) -> List[str]:
    """Return canonical values mentioned in text, longest aliases first."""
    found: List[str] = []
    lowered = text.lower()
    occupied: List[tuple[int, int]] = []
    for alias in sorted(aliases, key=len, reverse=True):
        for match in re.finditer(r'(?<![a-z0-9])' + re.escape(alias) + r'(?![a-z0-9])', lowered):
            if any(match.start() < end and match.end() > start for start, end in occupied):
                continue
            canonical = aliases[alias]
            if canonical not in found:
                found.append(canonical)
            occupied.append((match.start(), match.end()))
            break
    return found


def _fuzzy_aliases(text: str, aliases: dict[str, str]) -> List[str]:
    """Conservatively correct isolated misspellings of known query concepts."""
    words = WORD_RE.findall(text.lower())
    single_word_aliases = [a for a in aliases if ' ' not in a and len(a) >= 4]
    corrected: List[str] = []
    for word in words:
        if word in aliases or len(word) < 4:
            continue
        matches = difflib.get_close_matches(word, single_word_aliases, n=2, cutoff=0.86)
        if matches and (len(matches) == 1 or difflib.SequenceMatcher(None, word, matches[0]).ratio() -
                        difflib.SequenceMatcher(None, word, matches[1]).ratio() >= 0.04):
            corrected.append(matches[0])
    return corrected


def _expanded_query_tokens(text: str) -> List[str]:
    """Build lexical expansion tokens for one intent, not the whole request."""
    lowered = text.lower()
    expanded = list(WORD_RE.findall(lowered))
    fuzzy = _fuzzy_aliases(lowered, {**COLOR_SYNONYMS, **PRODUCT_TYPE_SYNONYMS})
    expanded.extend(fuzzy)
    for concept, terms in CONCEPT_EXPANSIONS.items():
        if concept in lowered:
            expanded.extend(terms)
    return list(dict.fromkeys(expanded))


def check_irrelevant(text_lower: str, words: set[str]) -> bool:
    """Check if query is non-fashion / out-of-domain and should be abstained on."""
    # Check if any catalog exception is present (e.g. mobile case)
    if any(re.search(r'(?<![a-z0-9])' + re.escape(exc) + r'(?![a-z0-9])', text_lower)
           for exc in CATALOG_ITEM_EXCEPTIONS):
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
    corrected_aliases = _fuzzy_aliases(text_lower, {**COLOR_SYNONYMS, **PRODUCT_TYPE_SYNONYMS, **AUDIENCE_SYNONYMS})
    recognition_text = text_lower + (' ' + ' '.join(corrected_aliases) if corrected_aliases else '')

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
    acc_price_min = None
    acc_price_max = None

    # Complementary pattern check
    # e.g. "A dress for dinner under INR 2000 with a matching bag under INR 1000"
    comp_match = re.match(
        r'^(?P<primary>.+?)\s+(?:with\s+(?:a\s+|an\s+)?(?:matching\s+)?|paired\s+with\s+(?:a\s+|an\s+)?|accompanied\s+by\s+(?:a\s+|an\s+)?|to\s+go\s+with\s+(?:a\s+|an\s+)?|along\s+with\s+(?:a\s+|an\s+)?|to\s+wear\s+with\s+(?:a\s+|an\s+)?|accessori[sz]e?d?\s+with\s+(?:a\s+|an\s+)?|complemented\s+by\s+(?:a\s+|an\s+)?|and\s+(?:a\s+|an\s+)?matching\s+)(?P<comp>.+)$',
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
            acc_price_min, acc_price_max = extract_budget(accessory_part)

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

    # 3. Extract Global budgets and facet mentions.  Explicit filters are
    # applied later and intentionally outrank these soft query hints.
    min_p, max_p = extract_budget(primary_part if intent_type == 'complementary' else query)

    # 4. Extract Colors
    detected_colors = _find_aliases(recognition_text, COLOR_SYNONYMS)

    # 5. Extract Audiences
    detected_audiences = _find_aliases(recognition_text, AUDIENCE_SYNONYMS)

    # 6. Extract Product Types
    detected_types = _find_aliases(recognition_text, PRODUCT_TYPE_SYNONYMS)

    accessory_colors = _find_aliases(accessory_part, COLOR_SYNONYMS) if accessory_part else []
    accessory_types = _find_aliases(accessory_part, PRODUCT_TYPE_SYNONYMS) if accessory_part else []

    # 7. Concept and Aesthetic Expansion
    expanded = _expanded_query_tokens(primary_part)

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
        accessory_price_min=acc_price_min,
        accessory_price_max=acc_price_max,
        colors=detected_colors,
        accessory_colors=accessory_colors,
        audiences=detected_audiences,
        product_types=detected_types,
        accessory_product_types=accessory_types,
        expanded_tokens=expanded
    )

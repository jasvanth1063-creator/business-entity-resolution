"""Phase 2 normalization registry.

Holds every candidate rule (name + address) in one place so the Phase 2
notebook can test them and the later phases can import the accepted subset.
The accepted subset is read from normalization_config.json, which the Phase 2
notebook writes after its rule tests. If the file is missing, the defaults
below (Phase 1 accepted rules only) are used.
"""
import json
import os
import re

from unidecode import unidecode

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_HERE, "normalization_config.json")

# --------------------------------------------------------------------------
# Name rules (fixed application order; config only chooses which are active)
# --------------------------------------------------------------------------
_DOMAIN_RE = re.compile(r"\.(?:com|net|org|co|in|biz|io)\b", re.IGNORECASE)


def _accent_strip(s):
    return unidecode(str(s))


def _lowercase(s):
    return s.lower()


def _ws(s):
    return re.sub(r"\s+", " ", s).strip()


def _suffix_canon(s):
    s = re.sub(r"\b(corporation)\b", "corp", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(private)\b", "pvt", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(limited)\b", "ltd", s, flags=re.IGNORECASE)
    return s


def _domain_strip(s):
    return _DOMAIN_RE.sub("", s)


def _ampersand(s):
    return s.replace("&", " and ")


def _name_punct(s):
    return re.sub(r"[^\w\s]", " ", s)


NAME_RULES = {
    "accent_strip": _accent_strip,
    "lowercase": _lowercase,
    "whitespace_collapse": _ws,
    "suffix_canonicalize": _suffix_canon,
    "domain_strip": _domain_strip,
    "ampersand_to_and": _ampersand,
    "punct_strip": _name_punct,
    "whitespace_final": _ws,
}
NAME_ORDER = ["accent_strip", "lowercase", "whitespace_collapse",
              "suffix_canonicalize", "domain_strip", "ampersand_to_and", "punct_strip",
              "whitespace_final"]
NAME_PHASE1_ACCEPTED = ["lowercase", "whitespace_collapse",
                        "suffix_canonicalize", "domain_strip", "whitespace_final"]

# --------------------------------------------------------------------------
# Address rules
# Order matters: landmark strip needs commas, abbreviation expansion needs
# dots/slashes (p.o. box, c/o), punctuation strip comes after both, and token
# sort comes last.
# --------------------------------------------------------------------------
ADDR_ABBR = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b|\bav\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bfl\b|\bflr\b": "floor",
    r"\bapt\b": "apartment",
    r"\bste\b": "suite",
    r"\bno\b|\bno\.\b": "number",
    r"\bc/o\b": "care of",
    r"\bpo box\b|\bp\.o\. box\b": "pobox",
}


def _addr_lower_ws(s):
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def _addr_landmark(s):
    s = re.sub(r"\b(?:near|opp|opposite|behind|beside|adjacent to)\b[^,]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _addr_abbrev(s):
    for pat, rep in ADDR_ABBR.items():
        s = re.sub(pat, rep, s)
    return s


def _addr_punct(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def _addr_token_sort(s):
    return " ".join(sorted(s.split()))


ADDR_RULES = {
    "landmark_strip": _addr_landmark,
    "abbrev_expand": _addr_abbrev,
    "punct_strip": _addr_punct,
    "token_sort": _addr_token_sort,
}
ADDR_ORDER = ["landmark_strip", "abbrev_expand", "punct_strip", "token_sort"]


# --------------------------------------------------------------------------
# Config + public API
# --------------------------------------------------------------------------
def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {"name_rules": list(NAME_PHASE1_ACCEPTED), "addr_rules": []}


def normalize_name(s, rules=None):
    rules = load_config()["name_rules"] if rules is None else rules
    s = str(s)
    for r in NAME_ORDER:
        if r in rules:
            s = NAME_RULES[r](s)
    return s


def normalize_addr(s, rules=None):
    rules = load_config()["addr_rules"] if rules is None else rules
    s = _addr_lower_ws(s)
    for r in ADDR_ORDER:
        if r in rules:
            s = ADDR_RULES[r](s)
    return s

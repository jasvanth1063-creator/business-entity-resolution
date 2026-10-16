"""
Shared utilities for Phase 1 - 4.
Combines core EDA functions, validated normalization rules,
address cleaning rules, vectorized key generation, and retrieval helpers.
"""
import re
import sys
import numpy as np
import pandas as pd
from collections import Counter
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jellyfish
from unidecode import unidecode
import eda_utils as _eda

# Re-export all eda_utils functions
NON_LATIN_RE = _eda.NON_LATIN_RE
DOMAIN_RE = _eda.DOMAIN_RE
SUFFIX_PATTERNS = _eda.SUFFIX_PATTERNS
chunked_noise_scan = _eda.chunked_noise_scan
inmemory_noise_scan = _eda.inmemory_noise_scan
token_jaccard = _eda.token_jaccard
edit_sim = _eda.edit_sim
pairwise_tfidf_cosine = _eda.pairwise_tfidf_cosine
rule_lowercase = _eda.rule_lowercase
rule_punct_strip = _eda.rule_punct_strip
rule_ampersand = _eda.rule_ampersand
rule_whitespace = _eda.rule_whitespace
rule_suffix_canon = _eda.rule_suffix_canon
rule_domain_strip = _eda.rule_domain_strip
DRAFT_RULES = _eda.DRAFT_RULES
apply_all_rules = _eda.apply_all_rules
key_first4 = _eda.key_first4
key_first_token = _eda.key_first_token
key_soundex_first_token = _eda.key_soundex_first_token
BLOCKING_KEYS = _eda.BLOCKING_KEYS
fetch_records = _eda.fetch_records

# Phase 2 Rules
def rule_strip_accents(s):
    return unidecode(str(s))

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
    r"\bpo box\b|\bp\.o\. box\b": "pobox"
}

def rule_addr_abbrev(s):
    s = str(s).lower()
    for pat, rep in ADDR_ABBR.items():
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)
    return s

def rule_landmark_strip(s):
    s = re.sub(r"\b(?:near|opp|opposite|behind|beside|adjacent to)\b[^,]+", " ", str(s), flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", s).strip()

ADDR_DRAFT_RULES = {
    "addr_abbrev_expand": rule_addr_abbrev,
    "landmark_strip": rule_landmark_strip,
}

# Vectorized normalization
def vectorized_normalize_name(series):
    s = series.astype(str).str.lower()
    s = s.str.replace(r"\.(?:com|net|org|co|in|biz|io)\b", "", regex=True)
    s = s.str.replace(r"\b(?:corporation)\b", "corp", regex=True)
    s = s.str.replace(r"\b(?:private)\b", "pvt", regex=True)
    s = s.str.replace(r"\b(?:limited)\b", "ltd", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

def vectorized_key_first4(series_norm, series_country):
    clean = series_norm.str.replace(r"[^a-z0-9]", "", regex=True)
    return clean.str[:4] + "_" + series_country.astype(str)

def vectorized_key_soundex(series_norm, series_country):
    def get_sx(val):
        toks = re.sub(r"[^a-z0-9\s]", " ", str(val)).split()
        t = toks[0] if toks else ""
        try:
            return jellyfish.soundex(t) if t else ""
        except Exception:
            return ""
    return series_norm.apply(get_sx) + "_" + series_country.astype(str)

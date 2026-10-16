"""
Phase 1 EDA — shared helper functions.
Kept separate from the notebook so both dev-testing (as a script) and the
final notebook cells import the exact same, already-verified logic.
"""
import re
import random
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jellyfish

# ---------------------------------------------------------------------------
# Noise-pattern regexes
# ---------------------------------------------------------------------------
NON_LATIN_RE = re.compile(r"[^\x00-\x7F]")
DOMAIN_RE = re.compile(r"\.(?:com|net|org|co|in|biz|io)\b", re.IGNORECASE)
SUFFIX_PATTERNS = {
    "corp/corporation": re.compile(r"\b(?:corp|corporation)\b", re.IGNORECASE),
    "pvt/private": re.compile(r"\b(?:pvt|private)\b", re.IGNORECASE),
    "ltd/limited": re.compile(r"\b(?:ltd|limited)\b", re.IGNORECASE),
    "inc": re.compile(r"\binc\b", re.IGNORECASE),
    "llc": re.compile(r"\bllc\b", re.IGNORECASE),
    "ampersand": re.compile(r"&"),
}

# ---------------------------------------------------------------------------
# Chunked full-file noise scan (for S2 / S3 — never loaded fully into memory)
# ---------------------------------------------------------------------------
def chunked_noise_scan(path, chunksize=250_000, sample_size=0, seed=42):
    """Streams a source tsv, accumulates noise-pattern counts + length sums,
    and (if sample_size>0) reservoir-samples ~sample_size random full rows
    in the same single pass, so a second full scan isn't needed later."""
    total = 0
    missing_addr = 0
    non_latin_name = 0
    domain_style = 0
    suffix_hits = {k: 0 for k in SUFFIX_PATTERNS}
    name_len_sum = 0
    addr_len_sum = 0
    country_counts = {}
    sample_rows = []
    rng = np.random.default_rng(seed)

    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                              chunksize=chunksize):
        n = len(chunk)
        total += n
        missing_addr += (chunk["business_address"].str.strip() == "").sum()
        non_latin_name += chunk["business_name"].str.contains(NON_LATIN_RE, regex=True).sum()
        domain_style += chunk["business_name"].str.contains(DOMAIN_RE, regex=True).sum()
        for k, pat in SUFFIX_PATTERNS.items():
            suffix_hits[k] += chunk["business_name"].str.contains(pat, regex=True).sum()
        name_len_sum += chunk["business_name"].str.len().sum()
        addr_len_sum += chunk["business_address"].str.len().sum()
        vc = chunk["country"].value_counts()
        for country, cnt in vc.items():
            country_counts[country] = country_counts.get(country, 0) + cnt
        if sample_size > 0:
            take = max(1, int(sample_size * n / 5_300_000))
            if take > 0 and n > 0:
                idx = rng.choice(n, size=min(take, n), replace=False)
                sample_rows.append(chunk.iloc[idx])

    sample_df = pd.concat(sample_rows, ignore_index=True) if sample_rows else pd.DataFrame()

    return {
        "total": total,
        "missing_addr_pct": 100 * missing_addr / total,
        "non_latin_name_pct": 100 * non_latin_name / total,
        "domain_style_pct": 100 * domain_style / total,
        "suffix_hits_pct": {k: 100 * v / total for k, v in suffix_hits.items()},
        "avg_name_len": name_len_sum / total,
        "avg_addr_len": addr_len_sum / total,
        "country_counts": country_counts,
        "sample_df": sample_df,
    }


def inmemory_noise_scan(df):
    """Same metrics as chunked_noise_scan but for a dataframe already in memory (S1)."""
    total = len(df)
    missing_addr = (df["business_address"].str.strip() == "").sum()
    non_latin_name = df["business_name"].str.contains(NON_LATIN_RE, regex=True).sum()
    domain_style = df["business_name"].str.contains(DOMAIN_RE, regex=True).sum()
    suffix_hits = {k: df["business_name"].str.contains(pat, regex=True).sum()
                   for k, pat in SUFFIX_PATTERNS.items()}
    return {
        "total": total,
        "missing_addr_pct": 100 * missing_addr / total,
        "non_latin_name_pct": 100 * non_latin_name / total,
        "domain_style_pct": 100 * domain_style / total,
        "suffix_hits_pct": {k: 100 * v / total for k, v in suffix_hits.items()},
        "avg_name_len": df["business_name"].str.len().mean(),
        "avg_addr_len": df["business_address"].str.len().mean(),
        "country_counts": df["country"].value_counts().to_dict(),
    }


# ---------------------------------------------------------------------------
# Similarity features
# ---------------------------------------------------------------------------
def token_jaccard(a, b):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa and not sb:
        return 0.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def edit_sim(a, b):
    if not a and not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0


def pairwise_tfidf_cosine(texts_a, texts_b):
    """Fits one shared TF-IDF space over both lists, returns row-wise cosine sim."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    all_texts = list(texts_a) + list(texts_b)
    mat = vec.fit_transform(all_texts)
    n = len(texts_a)
    mat_a, mat_b = mat[:n], mat[n:]
    sims = np.array([
        cosine_similarity(mat_a[i], mat_b[i])[0, 0] for i in range(n)
    ])
    return sims


# ---------------------------------------------------------------------------
# Normalization rules (candidates — validated in section 8)
# ---------------------------------------------------------------------------
def rule_lowercase(s):
    return s.lower()

def rule_punct_strip(s):
    return re.sub(r"[^\w\s]", " ", s)

def rule_ampersand(s):
    return s.replace("&", " and ")

def rule_whitespace(s):
    return re.sub(r"\s+", " ", s).strip()

def rule_suffix_canon(s):
    s = re.sub(r"\b(corporation)\b", "corp", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(private)\b", "pvt", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(limited)\b", "ltd", s, flags=re.IGNORECASE)
    return s

def rule_domain_strip(s):
    return DOMAIN_RE.sub("", s)

DRAFT_RULES = {
    "lowercase": rule_lowercase,
    "punct_strip": rule_punct_strip,
    "ampersand_to_and": rule_ampersand,
    "whitespace_collapse": rule_whitespace,
    "suffix_canonicalize": rule_suffix_canon,
    "domain_strip": rule_domain_strip,
}

def apply_all_rules(s, rules=DRAFT_RULES):
    for fn in rules.values():
        s = fn(s)
    return s


# ---------------------------------------------------------------------------
# Blocking keys (feasibility check — section 7)
# ---------------------------------------------------------------------------
def key_first4(name, country):
    n = re.sub(r"[^a-z0-9]", "", name.lower())
    return f"{n[:4]}_{country}"

def key_first_token(name, country):
    toks = re.sub(r"[^a-z0-9\s]", " ", name.lower()).split()
    t = toks[0] if toks else ""
    return f"{t}_{country}"

def key_soundex_first_token(name, country):
    toks = re.sub(r"[^a-z0-9\s]", " ", name.lower()).split()
    t = toks[0] if toks else ""
    try:
        code = jellyfish.soundex(t) if t else ""
    except Exception:
        code = ""
    return f"{code}_{country}"

BLOCKING_KEYS = {
    "first4_country": key_first4,
    "first_token_country": key_first_token,
    "soundex_first_token_country": key_soundex_first_token,
}


# ---------------------------------------------------------------------------
# Fetch specific entity_ids from a large source file via chunked scan
# ---------------------------------------------------------------------------
def fetch_records(path, wanted_ids, chunksize=250_000):
    """Returns {entity_id: {'business_name':..,'business_address':..,'country':..}}"""
    wanted = set(wanted_ids)
    found = {}
    if not wanted:
        return found
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                              chunksize=chunksize):
        hit = chunk[chunk["entity_id"].isin(wanted)]
        if len(hit):
            for row in hit.itertuples(index=False):
                found[row.entity_id] = {
                    "business_name": row.business_name,
                    "business_address": row.business_address,
                    "country": row.country,
                }
        if len(found) == len(wanted):
            break
    return found

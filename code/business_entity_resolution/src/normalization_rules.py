"""Validated normalization rules from Phase 1 EDA - accepted set only.
Accepted: ['lowercase', 'whitespace_collapse', 'suffix_canonicalize', 'domain_strip']
"""
import re

DOMAIN_RE = re.compile(r'\.(?:com|net|org|co|in|biz|io)\b', re.IGNORECASE)

def rule_lowercase(s):
    return s.lower()

def rule_whitespace(s):
    return re.sub(r"\s+", " ", s).strip()

def rule_suffix_canon(s):
    s = re.sub(r"\b(corporation)\b", "corp", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(private)\b", "pvt", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(limited)\b", "ltd", s, flags=re.IGNORECASE)
    return s

def rule_domain_strip(s):
    return DOMAIN_RE.sub("", s)

def normalize(s):
    s = rule_lowercase(s)
    s = rule_whitespace(s)
    s = rule_suffix_canon(s)
    s = rule_domain_strip(s)
    return s

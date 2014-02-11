from .levenshtein_cython import levenshtein_distance
from urlparse import urlparse

def block_length(anchor, _):
    tokens = anchor.text.split()
    if len(tokens) == 1:
        bl = '1'
    elif len(tokens) == 2:
        bl = '2'
    elif 2 < len(tokens) <= 10:
        bl = 'short'
    elif 10 < len(tokens) <= 20:
        bl = 'medium'
    else:
        bl = 'large'
    return {'block_length': bl}

def parent_tag(anchor, _):
    return {'parent_tag': anchor.getparent().tag}

def number_pattern(anchor, _):
    tokens = anchor.text.split()
    digits = [1 for token in tokens if token.isdigit()]
    if len(digits) == len(tokens):
        np = 'all'
    elif len(digits) >= 1:
        np = 'part'
    elif len(digits) == 0:
        np = 'no'
    return {'number_pattern': np}

def url_edit_distance(anchor, url):
    href = anchor.get('href', '')
    s1 = "".join(urlparse(href)[2:])
    s2 = "".join(urlparse(url)[2:])
    d = levenshtein_distance(s1, s2)
    # normalize
    return float(d) / max(len(s1), len(s2))

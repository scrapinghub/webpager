
def block_length(anchor):
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

def parent_tag(anchor):
    return {'parent_tag': anchor.getparent().tag}

def number_pattern(anchor):
    tokens = anchor.text.split()
    digits = [1 for token in tokens if token.isdigit()]
    if len(digits) == len(tokens):
        np = 'all'
    elif len(digits) >= 1:
        np = 'part'
    elif len(digits) == 0:
        np = 'no'
    return {'number_pattern': np}


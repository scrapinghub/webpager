import re

class Tagset(object):
    """
    Utility class for working with tags and converting between
    ``<TAG>`` and internal tag representation.
    """
    def __init__(self, tagset):
        self.tagset = tagset
        self.patterns = self._patterns()

    def encode_tags(self, text):
        """
        Replace <tag> and </tag> with __TAG_START and __TAG_END.

        This is needed to simplify parsing of HTML that has NER <tags>
        embedded (e.g. GATE outputs such HTML).

        >>> tagset = Tagset({'org', 'city'})
        >>> tagset.encode_tags('<p>Go to <CITY>Montevideo</city></p>')
        '<p>Go to  __CITY_START Montevideo __city_END </p>'

        """
        text = re.sub(self.patterns['html_open_tag'], r' __\1_START ', text)
        text = re.sub(self.patterns['html_close_tag'], r' __\1_END ', text)
        return text

    def start_tag_or_none(self, token):
        """
        >>> tagset = Tagset({'org'})
        >>> tagset.start_tag_or_none('foo')
        >>> tagset.start_tag_or_none('__ORG_START')
        'ORG'
        """
        if self.patterns['start_tag'].match(token):
            return token[2:-6].upper()

    def end_tag_or_none(self, token):
        """
        >>> tagset = Tagset({'org'})
        >>> tagset.start_tag_or_none('foo')
        >>> tagset.end_tag_or_none('__ORG_END')
        'ORG'
        """
        if self.patterns['end_tag'].match(token):
            return token[2:-4].upper()

    def _patterns(self):
        tags_pattern = '|'.join(self.tagset)
        return {
            'html_open_tag': re.compile('<(%s)>' % tags_pattern, re.I),
            'html_close_tag': re.compile('</(%s)>' % tags_pattern, re.I),
            'start_tag': re.compile('__(%s)_START' % tags_pattern, re.I),
            'end_tag': re.compile('__(%s)_END' % tags_pattern, re.I)
        }

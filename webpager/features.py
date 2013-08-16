import lxml
from lxml.html import tostring
from lxml.html.clean import Cleaner
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import scipy
import numpy as np

from .functions import parent_tag, block_length, number_pattern
from .preprocess import Tagset


_cleaner = Cleaner(
    style=True,
    scripts=True,
    embedded=True,
    links=True,
    page_structure=False,
    remove_unknown_tags=False,
    meta=False,
    safe_attrs_only=False
)

def tokenize(text):
    return text.split()

default_funcs = (parent_tag, block_length, number_pattern)

class HtmlFeaturesExtractor(BaseEstimator):
    """
    Extract the features for all the anchors from a HTML page.
    """
    def __init__(self, default_tags=('PAGE',), tokenize=tokenize):
        self.tagset = Tagset(default_tags)
        self.tokenize = tokenize

    @classmethod
    def clean_html(cls, html, encoding=None):
        parser = lxml.html.HTMLParser(encoding=encoding)

        if isinstance(html, unicode) and encoding is not None:
            html = html.encode(encoding)

        html = lxml.html.document_fromstring(html, parser=parser)
        return _cleaner.clean_html(html)

    def _parse_html(self, html, encoding=None):
        return self.clean_html(html, encoding)

    def fit_transform(self, X, y=None, encoding=None):
        """
        Convert the HTML data :param:X to list of the features.
        :param:y is ignored.
        """
        html = self.tagset.encode_tags(X)
        doc = self.clean_html(html, encoding)
        anchors = []
        labels = []
        for anchor in doc.iter('a'):
            tokens = self.tokenize(anchor.text or '')
            no_tag_tokens = [token for token in tokens if not (self.tagset.start_tag_or_none(token) or self.tagset.end_tag_or_none(token))]
            anchor.text = u" " .join(no_tag_tokens)
            anchors.append(anchor)
            labels.append(1 if len(tokens) != len(no_tag_tokens) else 0)

        return anchors, labels

class AnchorContextTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_funcs):
        self.feature_funcs = feature_funcs
        self.dict_vectorizer = DictVectorizer()

    def get_feature_names(self):
        return self.dict_vectorizer.get_feature_names()

    def fit_transform(self, X, y=None):
        return self.dict_vectorizer.fit_transform(self._apply_funcs(x) for x in X)

    def transform(self, X):
        return self.dict_vectorizer.transform(self._apply_funcs(x) for x in X)

    def _apply_funcs(self, x):
        d = {}
        for func in self.feature_funcs:
            d.update(func(x))
        return d


class AnchorFeatureTransformer(BaseEstimator, TransformerMixin):

    """
    Extract the features from anchors
    """

    def __init__(self, feature_funcs=default_funcs):
        self.feature_funcs = feature_funcs
        self._text_vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 5), min_df=1, binary=True)
        self._class_vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 5), min_df=1, binary=True)
        self._misc_vectorizer = AnchorContextTransformer(feature_funcs)

    def get_feature_names(self):
        feature_names = []
        feature_names.extend(self._text_vectorizer.get_feature_names())
        feature_names.extend(self._class_vectorizer.get_feature_names())
        feature_names.extend(self._misc_vectorizer.get_feature_names())
        return np.array(feature_names)

    def fit_transform(self, X, y=None):
        _texts = [anchor.text for anchor in X]
        _classes = [anchor.get('class', '') + anchor.get('id', '') for anchor in X]
        texts_grams = self._text_vectorizer.fit_transform(_texts)
        _class_grams = self._class_vectorizer.fit_transform(_classes)
        _misc_features = self._misc_vectorizer.fit_transform(X)

        features = []
        features.append(texts_grams)
        features.append(_class_grams)
        features.append(_misc_features)
        features = scipy.sparse.hstack(features).tocsr()

        return features

    def transform(self, X):
        _texts = [anchor.text for anchor in X]
        _classes = [anchor.get('class', '') + anchor.get('id', '') for anchor in X]

        texts_grams = self._text_vectorizer.transform(_texts)
        _class_grams = self._class_vectorizer.transform(_classes)
        _misc_features = self._misc_vectorizer.transform(X)

        features = []
        features.append(texts_grams)
        features.append(_class_grams)
        features.append(_misc_features)
        features = scipy.sparse.hstack(features).tocsr()

        return features

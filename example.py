"""
Usage: python example.py URL

it will show all the links and predicted label (1 or pagination, 0 otherwise) afterwards.

"""
from urllib2 import urlopen
from urlparse import urljoin
import sys
from w3lib.encoding import html_to_unicode
from webpager.models import get_models

htmlFeatGen, anchorFeatGen, clf = get_models()
url = sys.argv[1]
html = urlopen(url).read()
_, html = html_to_unicode(None, html)
anchors, _ = htmlFeatGen.fit_transform(html)
documents = anchorFeatGen.transform(anchors)
labels = clf.predict(documents)

for anchor, label in zip(anchors, labels):
    print urljoin(url, anchor.get('href')), label
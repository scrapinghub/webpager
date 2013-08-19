"""
Usage: python example.py URL

it will show all the links and predicted label (1 or pagination, 0 otherwise) afterwards.

"""
from urllib2 import urlopen
from urlparse import urljoin
from sklearn.externals import joblib
import sys
from w3lib.encoding import html_to_unicode

url = sys.argv[1]

htmlFeatGen = joblib.load('htmlFeatGen.joblib.pkl')
anchorFeatGen = joblib.load('anchorFeatGen.joblib.pkl')
clf = joblib.load('clf.joblib.pkl')

html = urlopen(url).read()
_, html = html_to_unicode(None, html)
anchors, _ = htmlFeatGen.fit_transform(html)
documents = anchorFeatGen.transform(anchors)
labels = clf.predict(documents)

for anchor, label in zip(anchors, labels):
    print urljoin(url, anchor.get('href')), label
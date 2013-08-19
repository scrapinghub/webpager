"""
Usage: python example.py URL

it will show all the links and predicted label (1 or pagination, 0 otherwise) afterwards.

"""
from urllib2 import urlopen
from urlparse import urljoin
from sklearn.externals import joblib
import sys
from w3lib.encoding import html_to_unicode

htmlFeatGen = joblib.load('htmlFeatGen.joblib.pkl')
anchorFeatGen = joblib.load('anchorFeatGen.joblib.pkl')
clf = joblib.load('clf.joblib.pkl')
url = sys.argv[1]

html = urlopen(url).read()
_, html = html_to_unicode(None, html)
anchors, _ = htmlFeatGen.fit_transform(html)
documents = anchorFeatGen.transform(anchors)

for i, document in enumerate(documents):
    print urljoin(url, anchors[i].get('href')), clf.predict(document)[0]
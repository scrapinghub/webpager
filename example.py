"""
Usage: python example.py URL

it will show all the links and predicted label (1 or pagination, 0 otherwise) afterwards.

"""
from urllib2 import urlopen
from urlparse import urljoin
from sklearn.externals import joblib
import sys
from w3lib.encoding import html_to_unicode

htmlGen = joblib.load('htmlGen.joblib.pkl')
anchorGen = joblib.load('anchorGen.joblib.pkl')
clf = joblib.load('clf.joblib.pkl')
url = sys.argv[1]

html = urlopen(url).read()
_, html = html_to_unicode(None, html)
anchors, _ = htmlGen.fit_transform(html)
documents = anchorGen.transform(anchors)

for i, document in enumerate(documents):
    print urljoin(url, anchors[i].get('href')), clf.predict(document)[0]
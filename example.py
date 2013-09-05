"""
Usage: python example.py URL

it will show all the links and predicted label (1 or pagination, 0 otherwise) afterwards.

"""
from urllib2 import urlopen
from urlparse import urljoin
import sys
from w3lib.encoding import html_to_unicode
from webpager import WebPager

webpager = WebPager()
url = sys.argv[1]
html = urlopen(url).read()
_, html = html_to_unicode(None, html)

for anchor, label in webpager.paginate(html):
    print urljoin(url, anchor.get('href')), label
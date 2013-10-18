Webpager
========

A simple library to classify if an anchor on HTML page is a pagination link or not.

Installation
========

Clone the repository, then install package requirements
(package requires lxml, scikit-learn)

    $ pip install -r requirements.txt

then install package itself

    $ python setup.py install

Usage
========
Get a HTML page somewhere.

```python
>>> from urlparse import urljoin
>>> html = urlopen(url).read()
```
Load web pager and classify.

```python
>>> from webpager import WebPager
>>> from urllib2 import urlopen
>>> webpager = WebPager()
>>> for anchor, label in webpager.paginate(html):
>>>     if label:
>>>	         print urljoin(url, anchor.get('href'))
```

Training
========
see `train.py` for more details.
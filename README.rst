Webpager
========

A simple library to classify if an anchor on HTML page is a pagination link or not.

Installation
========

Clone the repository, then install package requirements
(package requires lxml, scikit-learn)::

    $ pip install -r requirements.txt

then install package itself::

    $ python setup.py install

Usage
========
Get a HTML page somewhere.::

    >>> from urllib import urlopen
    >>> url = 'http://www.tripadvisor.com/Restaurant_Review-g294217-d3639657-Reviews-Trattoria_Caffe_Monteverdi-Hong_Kong.html'
    >>> html = urlopen(url).read()

Load web pager and classify.::

    >>> from webpager import WebPager
    >>> webpager = WebPager()
    >>> for anchor, label in webpager.paginate(html, url):
    >>>     if label:
    >>>	         print urljoin(url, anchor.get('href'))

    http://www.tripadvisor.com/Restaurant_Review-g294217-d3639657-Reviews-or10-Trattoria_Caffe_Monteverdi-Hong_Kong.html#REVIEWS
    http://www.tripadvisor.com/Restaurant_Review-g294217-d3639657-Reviews-or40-Trattoria_Caffe_Monteverdi-Hong_Kong.html#REVIEWS
    http://www.tripadvisor.com/Restaurant_Review-g294217-d3639657-Reviews-or10-Trattoria_Caffe_Monteverdi-Hong_Kong.html#REVIEWS


Training
========
see `train.py` for more details.
from .models import get_models
from .features import HtmlFeaturesExtractor

class WebPager(object):
    def __init__(self):
        self.anchorFeatGen, self.clf = get_models()
        self.htmlFeatGen = HtmlFeaturesExtractor()

    def paginate(self, html, base_url):
        anchors, _ = self.htmlFeatGen.fit_transform(html, base_url)
        documents = self.anchorFeatGen.transform([(anchor, base_url) for anchor in anchors])
        labels = self.clf.predict(documents)

        return zip(anchors, labels)
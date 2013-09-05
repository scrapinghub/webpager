from .models import get_models


class WebPager(object):
    def __init__(self):
        self.htmlFeatGen, self.anchorFeatGen, self.clf = get_models()

    def paginate(self, html):
        anchors, _ = self.htmlFeatGen.fit_transform(html)
        documents = self.anchorFeatGen.transform(anchors)
        labels = self.clf.predict(documents)

        pages = []
        for anchor, label in zip(anchors, labels):
            pages.append((anchor, label))

        return pages
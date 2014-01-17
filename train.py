import os
from os.path import join
import posixpath

from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.cross_validation import train_test_split

from webpager.features import HtmlFeaturesExtractor, AnchorTransformers

def tagged_data(folder):
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        with open(path, 'r') as f:
            yield path, f.read()

def get_original_urls(fname):
    return dict([str(i+1), line.strip()] for i, line in enumerate(open(fname)))

if __name__ == '__main__':

    htmlFeatGen = HtmlFeaturesExtractor()
    anchorFeatGen = FeatureUnion(AnchorTransformers)
    base_urls = get_original_urls('corpus/list.txt')
    documents = []
    labels = []

    for path, html in tagged_data('corpus/annotated'):
        _id = posixpath.split(path)[1].split('.')[0]
        anchors_, labels_ = htmlFeatGen.fit_transform(html, baseurl=base_urls[_id], encoding='utf8')
        documents.extend([(a, base_urls[_id]) for a in anchors_])
        labels.extend(labels_)

    train_anchors, test_anchors, train_labels, test_labels = train_test_split(documents, labels, \
                                                                              test_size=0.25, random_state=42)
    train_documents = anchorFeatGen.fit_transform(train_anchors, y=train_labels)
    test_documents = anchorFeatGen.transform(test_anchors)

    clf = LogisticRegression(tol=1e-8, penalty='l2', C=7, class_weight='auto')
    clf.fit(train_documents, train_labels)

    predicted = clf.predict(test_documents)
    print '# train documents', train_documents.shape[0]
    print '# test documents', test_documents.shape[0]
    print 'confusion matrix:'
    print confusion_matrix(test_labels, predicted)
    print 'accuracy', accuracy_score(test_labels, predicted)
    print 'precision', precision_score(test_labels, predicted)
    print 'recall', recall_score(test_labels, predicted)

    def show_most_informative_features(vectorizer, clf, n=20):
        c_f = sorted(zip(clf.coef_[0], vectorizer.get_feature_names()))
        top = zip(c_f[:n], c_f[:-(n+1):-1])
        for (c1,f1),(c2,f2) in top:
            print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (c1,f1,c2,f2)

    show_most_informative_features(anchorFeatGen, clf)

    joblib.dump(anchorFeatGen, join('webpager', 'models', 'anchorFeatGen.joblib.pkl'), compress=9)
    joblib.dump(clf, join('webpager', 'models', 'clf.joblib.pkl'), compress=9)

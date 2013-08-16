import os
from sklearn.dummy import DummyClassifier
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score
from sklearn.linear_model import LogisticRegression
import numpy as np

from webpager.features import HtmlFeaturesExtractor, AnchorFeatureTransformer


htmlGen = HtmlFeaturesExtractor()
anchorGen = AnchorFeatureTransformer()

def tagged_data(folder):
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        with open(path, 'r') as f:
            yield path, f.read()

anchors = []
labels = []

for _, html in tagged_data('corpus/annotated'):
    anchor, label = htmlGen.fit_transform(html, encoding='utf8')
    anchors.extend(anchor)
    labels.extend(label)

documents = anchorGen.fit_transform(anchors[:3000])
test_documents = anchorGen.transform(anchors[3000:])

baseclf = DummyClassifier(strategy='most_frequent',random_state=0)
baseclf.fit(documents, labels[:3000])

target_names = ['NOT-PAGE', 'PAGE']

clf = LogisticRegression(tol=1e-8, penalty='l2', C=7, class_weight='auto')
clf.fit(documents, labels[:3000])

predicted = clf.predict(test_documents)
print confusion_matrix(labels[3000:], predicted)
print 'accuracy', accuracy_score(labels[3000:], predicted)
print 'recall', recall_score(labels[3000:], predicted)

print '-' * 80

base_predicted = baseclf.predict(test_documents)
print confusion_matrix(labels[3000:], base_predicted)
print 'accuracy', accuracy_score(labels[3000:], base_predicted)
print 'recall', recall_score(labels[3000:], base_predicted)

def print_top10(vectorizer, clf, n=20):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    top10 = np.argsort(clf.coef_[0])[-n:]
    print(" ".join(feature_names[j] for j in top10))

print '-' * 80
print_top10(anchorGen, clf)

joblib.dump(htmlGen, 'htmlGen.joblib.pkl', compress=9)
joblib.dump(anchorGen, 'anchorGen.joblib.pkl', compress=9)
joblib.dump(clf, 'clf.joblib.pkl', compress=9)
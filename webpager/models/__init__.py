from sklearn.externals import joblib
from os.path import dirname, join

def get_models():
    htmlFeatGen = joblib.load(join(dirname(__file__), 'htmlFeatGen.joblib.pkl'))
    anchorFeatGen = joblib.load(join(dirname(__file__), 'anchorFeatGen.joblib.pkl'))
    clf = joblib.load(join(dirname(__file__), 'clf.joblib.pkl'))
    return htmlFeatGen, anchorFeatGen, clf

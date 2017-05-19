from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

class Trainer():
    @classmethod
    def run(cls, X, y):
        estimator = LogisticRegression(penalty='l2', C=0.01)
        estimator.fit(X, y)
        joblib.dump(estimator, './model/lr.pkl')

    @classmethod
    def lowest_highest_features(cls):
        estimator = joblib.load('./model/lr.pkl')
        lowest = np.argsort(estimator.coef_[0])[:10]
        highest = np.argsort(estimator.coef_[0])[::-1][:10]
        dictionary = Preprocessor.load_dictionray()
        highest_feature_names = [list(dictionary.token2id.keys())[list(dictionary.token2id.values()).index(idx)] for idx in list(highest)]
        lowest_feature_names = [list(dictionary.token2id.keys())[list(dictionary.token2id.values()).index(idx)] for idx in list(lowest)]
        return [highest_feature_names, lowest_feature_names]

from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
import numpy as np

class Predictor:
    estimator = joblib.load('./model/lr.pkl')

    @classmethod
    def predict(cls, docs):
        dense = Preprocessor.create_dense(docs)
        results = cls.estimator.predict(dense)
        return results

    @classmethod
    def predict_proba(cls, docs):
        dense = Preprocessor.create_dense(docs)
        results = cls.estimator.predict(dense)
        probas = cls.estimator.predict_proba(dense)
        return [results, probas]

    @classmethod
    def predict_proba_by_dense(cls, dense):
        results = cls.estimator.predict(dense)
        probas = cls.estimator.predict_proba(dense)
        return [results, probas]

    @classmethod
    def f1_score(cls, predicts, corrects):
        results = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

        for predict, correct in zip(predicts, corrects):
            if predict == '+1' and correct == '+1':
                results['tp'] += 1
            elif predict == '-1' and correct == '-1':
                results['tn'] += 1
            elif predict == '+1' and correct == '-1':
                results['fp'] += 1
            elif predict == '-1' and correct == '+1':
                results['fn'] += 1

        acc = (results['tp'] + results['tn']) / len(predicts)
        pre = results['tp'] / (results['tp'] + results['fp'])
        rec = results['tp'] / (results['tp'] + results['fn'])
        f1 = 2 * (pre * rec) / (pre + rec)
        return [acc, pre, rec, f1]

    @classmethod
    def cross_validation(cls, X, y):
        hist = {'acc': [], 'pre': [], 'rec': [], 'f1': []}
        for i in range(5):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            Trainer.run(X_train, y_train)
            predicts = cls.estimator.predict(X_test)
            acc, pre, rec, f1 = cls.f1_score(predicts, y_test)
            print('[{}回] 正解率: {} 適合率: {} 再現率: {} F1スコア: {}'.format(i + 1, acc, pre, rec, f1))
            hist['acc'].append(acc)
            hist['pre'].append(pre)
            hist['rec'].append(rec)
            hist['f1'].append(f1)
        return [np.mean(hist['acc']), np.mean(hist['pre']), np.mean(hist['rec']), np.mean(hist['f1'])]



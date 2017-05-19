from nltk import stem
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
import seaborn as sns



class Util:
    stop_words = [line.rstrip() for line in open('stopwords.txt').readlines()]

    @staticmethod
    def is_stop_word(word):
        return word.lower() in Util.stop_words

    @staticmethod
    def normalize_word(word):
        word = word.lower()
        stemmer = stem.PorterStemmer()
        return stemmer.stem(word)

    @staticmethod
    def is_invalid_word(word):
        return len(word) < 2 or Util.is_stop_word(word)

    @staticmethod
    def plot_learning_curve(X, y):
        plt.rc('figure', figsize=(12, 8))

        estimator = LogisticRegression(penalty='l2', C=0.01)
        training_sizes, train_scores, test_scores = learning_curve(estimator=estimator, X=X, y=y, \
                                                                   train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0], cv=10)

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.plot(training_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
        plt.fill_between(training_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

        plt.plot(training_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5,
                 label='validation accuracy')
        plt.fill_between(training_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

        plt.grid()
        plt.xlabel('Number of training samples')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        plt.ylim(0.5, 1.0)

    @staticmethod
    def plot_validation_curve(X, y):
        param_range = [0.001, 0.01, 1.0, 10.0]
        estimator = LogisticRegression(penalty='l2')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        train_scores, test_scores = validation_curve(estimator=estimator, X=X_train, y=y_train, param_name='C',
                                                     param_range=param_range, cv=10)
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
        plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

        plt.plot(param_range, test_mean, color='blue', marker='o', markersize=5, label='validation accuracy')
        plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

        plt.grid()
        plt.xscale('log')
        plt.legend(loc='best')
        plt.xticks(param_range)
        plt.xlabel('Parameter C')
        plt.ylabel('Accuracy')

    @staticmethod
    def plot_precision_recall_curve(X, y):
        plt.rc('figure', figsize=(12, 8))
        results, probas = Predictor.predict_proba_by_dense(X)
        predictions = probas[:, 0]
        yi = [1 if u == '+1' else -1 for u in y]

        precision, recall, thresholds = precision_recall_curve(yi, predictions)
        thresholds = np.append(thresholds, 1)

        plt.plot(thresholds, precision, color=sns.color_palette()[0])
        plt.plot(thresholds, recall, color=sns.color_palette()[1])

        plt.legend(('precision', 'recall'), loc='best')
        plt.xlabel('threshold')
        plt.ylabel('%')
        plt.ylim(0.0, 1.01)



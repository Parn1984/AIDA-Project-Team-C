"""
+++ AUTHOR: Tobias Habermann +++

KNN classification class

Methods:
    set_train : set training data
    set_test : set test data
    fit : train KNN model with given training data
    predict : predict data on test set and write it to class variable y_pred
    get_cross_validation_accuracy : calculates cross vaidation score for given training data
    get_test_accuracy : returns accuracy of prediction (y_pred) and given test data
    tune_parameters : tunes k-neighbors and weight of KNN model and uses the best parameters
    print_tuning_results : prints best parameters calculated in tune_parameters() method
    plot_tuning_k_results : plots k and corresponding score calculated in tune_parameters() method
    plot_confusion_matrix : plot confusion matrix
"""


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Class for KNN classification

class KNN_classification():
    KNNmodel = KNeighborsClassifier()
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    y_pred = None
    accuracy = None
    cv_folds = 2  # due to given tasks two fold cross validation

    def __init__(self):
        print('KNN Classification created')

    # Set trainings set
    def set_train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # Set test set
    def set_test(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    # fit the model
    def fit(self):
        self.KNNmodel.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_pred = self.KNNmodel.predict(self.X_test)

    def get_cross_validation_accuracy(self):
        self.cv_score = cross_val_score(self.KNNmodel, self.X_train, self.y_train, cv=self.cv_folds)
        return self.cv_score.mean()

    def get_test_accuracy(self):
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        return self.accuracy

    # find best weight and number of neighbors
    def tune_parameters(self):

        # it is good practice to use sqrt of features as k
        sqrt_k = int(round(len(self.X_train) ** 0.5, 0))
        print('Square-root of number of datasets:', sqrt_k)

        self.dict_KNN = dict()

        self.best_score = 0.0

        for k in range(1, sqrt_k + 1):  # we check up to the best practice value ;-)

            for weight in ['uniform', 'distance']:
                # ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
                # ‘distance’ : weight points by the inverse of their distance.
                #              in this case, closer neighbors of a query point will have a greater
                #              influence than neighbors which are further away.

                tuneKNNmodel = KNeighborsClassifier(n_neighbors=k, weights=weight)
                tuneKNNmodel.fit(self.X_train, self.y_train)
                KNNscores = cross_val_score(tuneKNNmodel, self.X_train, self.y_train, cv=self.cv_folds)

                score = KNNscores.mean()

                self.dict_KNN[k] = KNNscores.mean()

                # sore best values and model
                if score > self.best_score:
                    self.best_k = k
                    self.best_weight = weight
                    self.best_score = score
                    self.KNNmodel = tuneKNNmodel

        # after final tuning also update predicted values
        self.predict()

    # print cv scores and parameters collected during parameter tuning
    def print_tuning_results(self):
        print(f"Best Score: {self.best_score}\nBest k: {self.best_k}\nBest Weights: {self.best_weight}")

    # plot k and corresponding scores collected during parameter tunig
    def plot_tuning_k_results(self, figsize=(10, 5)):
        plt.figure(figsize=figsize)
        plt.plot(list(self.dict_KNN.keys()), list(self.dict_KNN.values()))
        plt.scatter(self.best_k, self.best_score, c='red', s=50)
        plt.axhline(self.best_score, color='red', linestyle='-.', linewidth=1)
        plt.axvline(self.best_k, color='red', linestyle='-.', linewidth=1)
        plt.xlabel("k-neighbors", fontsize=12)
        plt.ylabel("Accuracy score", fontsize=12)
        plt.title("Tuning k and weight for KNN", fontsize=14)
        plt.grid(color='lightgray')
        plt.show()

    def plot_confusion_matrix(self):

        # print dtaframe
        conf_matrix = pd.DataFrame(confusion_matrix(self.y_test, self.y_pred, normalize='true'),
                                   columns=['churn', 'no churn'], index=['churn', 'no churn'])
        conf_matrix.index.name = "True Label"
        conf_matrix.columns.name = "Predicted Label"
        display(conf_matrix)

        # plot
        # sns.heatmap(conf_matrix, annot=True, cmap='Greens')
        data = confusion_matrix(self.y_test, self.y_pred)
        df_cm = pd.DataFrame(data, columns=['no churn', 'churn'], index=['no churn', 'churn'])
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'
        plt.figure(figsize=(9, 6))
        sns.set(font_scale=1.4)  # for label size
        sns.heatmap(df_cm, cmap='RdYlGn', annot=True, annot_kws={"size": 14}, fmt='d')
        plt.title("Confusion Matrix", fontweight='bold')
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.show()

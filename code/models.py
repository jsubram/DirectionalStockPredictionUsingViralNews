import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class Models:

    def __init__(self, base_path, ticker, stock_price, timeframe):
        self.pickle_path = base_path + '/pickle/'
        self.stock_ticker = ticker
        self.stock_price = stock_price
        self.time_frame = timeframe
        self.train_data = pd.read_csv(self.pickle_path + self.stock_ticker + '_traindata_' + self.time_frame + '.csv',
                                      sep=',')
        self.x = self.train_data.loc[:, self.train_data.columns != 'Label'].values
        self.y = self.train_data['Label'].values
        self.stock_price = self.stock_price.reset_index(drop=True)
        self.x_train = self.x[:len(self.stock_price) * 60 // 100]
        self.x_test = self.x[len(self.stock_price) * 60 // 100:]
        self.y_train = self.y[:len(self.stock_price) * 60 // 100]
        self.y_test = self.y[len(self.stock_price) * 60 // 100:]

    def preprocess(self):
        self.train_data = self.train_data.astype(float)
        self.train_data = self.train_data.dropna()
        x = self.train_data.loc[:, self.train_data.columns != 'Label'].values
        y = self.train_data['Label'].values
        self.stock_price = self.stock_price.reset_index(drop=True)
        x_train = x[:len(self.stock_price) * 60 // 100]
        x_test = x[len(self.stock_price) * 60 // 100:]
        y_train = y[:len(self.stock_price) * 60 // 100]
        y_test = y[len(self.stock_price) * 60 // 100:]
        return x_train, x_test, y_train, y_test

    def naive_bayes_classifier(self):
        x_train, x_test, y_train, y_test = self.preprocess()
        clfnb = GaussianNB()
        clfnb.fit(x_train, y_train)
        print("Naive Bayes classifier")
        print(clfnb.score(x_test, y_test))

    # SVM classifier
    def SVM_classifier(self):
        x_train, x_test, y_train, y_test = self.preprocess()
        clf = svm.SVC()
        clf.fit(x_train, y_train)
        print("SVM rbf kernel Classifier")
        print(clf.score(x_test, y_test))

    # Decision Tree Classifier
    def DT_classifier(self):
        x_train, x_test, y_train, y_test = self.preprocess()
        clftree = tree.DecisionTreeClassifier()
        clftree.fit(x_train, y_train)
        print("Decision Tree Classifier")
        print(clftree.score(x_test, y_test))

        # SVM polynomial classifier

    def SVM_poly_classifier(self):
        x_train, x_test, y_train, y_test = self.preprocess()
        clf = svm.SVC(kernel='poly')
        clf.fit(x_train, y_train)
        print("SVM polynomial kernel Classifier")
        print(clf.score(x_test, y_test))

        # Logistic Regression l1 classifier

    def Logistic_Regression11_classifier(self):
        x_train, x_test, y_train, y_test = self.preprocess()
        clfl1 = LogisticRegression(penalty='l1')
        clfl1.fit(x_train, y_train)
        print("Logistic Regression l1 type classifier")
        print(clfl1.score(x_test, y_test))

        # Logistic Regression l2 classifier

    def Logistic_Regression12_classifier(self):
        x_train, x_test, y_train, y_test = self.preprocess()
        clfl2 = LogisticRegression(penalty='l2')
        clfl2.fit(x_train, y_train)
        print("Logistic Regression l2 type classifier")
        print(clfl2.score(x_test, y_test))

        # KneighboursClassifier

    def KNN_classifier(self):
        x_train, x_test, y_train, y_test = self.preprocess()
        clf = KNeighborsClassifier()
        clf.fit(x_train, y_train)
        print("KNeighborsClassifier")
        print(clf.score(x_test, y_test))

    # SGDClassifier
    def SGDC_classifier(self):
        x_train, x_test, y_train, y_test = self.preprocess()
        clf = SGDClassifier()
        clf.fit(x_train, y_train)
        print("SGDClassifier")
        print(clf.score(x_test, y_test))

    def accounting_factor(self):
        test_set = self.stock_price[len(self.stock_price) * 60 // 100:]
        test_set_profit = test_set[test_set['Direction']==1]

        print()

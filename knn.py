import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from collections import Counter
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

def knn():
    #Import and preprocessing
    dataset = pd.read_csv('_dataset.csv', error_bad_lines=False)
    X_train = dataset.iloc[:, :-1].values
    y_train = dataset.iloc[:, 7].values
    X_test = genfromtxt('_test.csv', delimiter=',', skip_header=1)

    #Feature Scaling
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    #Training and Prediction
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return Counter(y_pred).most_common()[0][0]

def testKnn():
    #Import and preprocessing
    dataset = pd.read_csv('_dataset.csv', error_bad_lines=False)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 7].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    #Feature Scaling
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    #Training and Prediction
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    testKnn()
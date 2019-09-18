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

def graphKnn():
    dataset = pd.read_csv('_dataset.csv', error_bad_lines=False)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 7].values
    h = .02
    print(X)

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA','#00AAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00','#00AAFF'])

    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X, y)

    # calculate min, max and limits
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h))

    # predict class using data and kNN classifier
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i)" % (n_neighbors))
    plt.show()

if __name__ == "__main__":
    graphKnn()  
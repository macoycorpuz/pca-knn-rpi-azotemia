import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# n_neighbors = 6

# # import some data to play with
# iris = datasets.load_iris()

# # prepare data
# X = iris.data[:, :2]
# y = iris.target
# h = .02
# print(X)

# # Create color maps
# cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA','#00AAFF'])
# cmap_bold = ListedColormap(['#FF0000', '#00FF00','#00AAFF'])

# # we create an instance of Neighbours Classifier and fit the data.
# clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
# clf.fit(X, y)

# # calculate min, max and limits
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
# np.arange(y_min, y_max, h))

# # predict class using data and kNN classifier
# Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure()
# plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# # Plot also the training points
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.title("3-Class classification (k = %i)" % (n_neighbors))
# plt.show()


sensors = ['MQ2','MQ3','MQ4','MQ6','MQ7','MQ8','MQ135']

df = pd.read_csv('_dataset.csv', error_bad_lines=False)
sensor_values = df.loc[:, sensors].values

sensor_values = StandardScaler().fit_transform(sensor_values)
pca = PCA(n_components=2)
components = pca.fit_transform(sensor_values)
principalDf = pd.DataFrame(data=components, columns=['pc1', 'pc2'])
finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

dataset = pd.read_csv('_dataset.csv', error_bad_lines=False)
X = principalDf.values
y = dataset.iloc[:, 7].values
h = .02

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

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
plt.scatter(X[:, 0], X[:, 1], cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = %i)" % (5))
plt.show()
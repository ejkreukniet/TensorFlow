# -*- coding: utf-8 -*-
from sklearn import svm, datasets
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from itertools import cycle

import matplotlib.pyplot as plt

import numpy as np
import pylab as pl

iris = load_iris()

numSamples, numFeatures = iris.data.shape
print(numSamples)
print(numFeatures)
print(list(iris.target_names))

X = iris.data
pca = PCA(n_components=2, whiten=True).fit(X)
X_pca = pca.transform(X)

print(pca.components_)

print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))

X = []
y = []

colors = cycle('rgb')
target_ids = range(len(iris.target_names))
for i, c, label in zip(target_ids, colors, iris.target_names):
    n = list(iris.target_names).index(label)
    for j in X_pca[iris.target == i, :]:
        X.append(j)
        y.append(n)

X = np.array(X)
y = np.array(y)
    
C = 1.0
# kernel = linear, poly, rbf, sigmoid, precomputed
svc = svm.SVC(kernel='linear', degree=3, C=C, gamma='auto').fit(X, y) 

xx, yy = np.meshgrid(np.arange(-2.5, 2.5, 0.1), np.arange(-3.5, 3.5, 0.1))
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])

#plt.figure(figsize=(8, 6))
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:,0], X[:,1], c=y.astype(np.float))

plt.show()
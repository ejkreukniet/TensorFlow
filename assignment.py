from keras.datasets import mnist
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from multiprocessing import Process

import pandas as pd

import pydotplus

from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.six import StringIO
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from tensorflow import keras

def score(p):
    correct = 0
    for i in range(208):
        if (predicted[i] == y_test[i]):
            correct += 1
    return correct/208

# This data contains 961 instances of masses detected in mammograms, and contains the following attributes:
#
#     BI-RADS assessment: 1 to 5 (ordinal)
#     Age: patient's age in years (integer)
#     Shape: mass shape: round=1 oval=2 lobular=3 irregular=4 (nominal)
#     Margin: mass margin: circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5 (nominal)
#     Density: mass density high=1 iso=2 low=3 fat-containing=4 (ordinal)
#     Severity: benign=0 or malignant=1 (binominal)

feature_names =  ['BI-RADS', 'Age', 'Shape', 'Margin', 'Density', 'Severity']

print('Loading data...')
df = pd.read_csv('~/Documents/mammographic_masses.data.txt', na_values=['?'], names = feature_names)
print('Shape data: ', df.shape)
#df.head()

#df.describe()

df.dropna(inplace=True)
print('After cleanup: ', df.shape)

scale = StandardScaler()
features = list(df.columns[1:5])
print('Features: ', features)
X = df[features]
y = df[df.columns[5:]].values
X = scale.fit_transform(X)
print('Shape X: ', X.shape)
print('Shape y: ', y.shape)
#print(X)

(X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=1)
print('Shape X_train: ', X_train.shape)
print('Shape X_test: ', X_test.shape)

dtc = tree.DecisionTreeClassifier(random_state=1)
dtc = dtc.fit(X_train, y_train)
print('\nDecisiontree score: ' , dtc.score(X_test, y_test))

def plot_graph():
	dot_data = StringIO()
	tree.export_graphviz(dtc, out_file=dot_data, feature_names=features, filled=True, rounded=True, special_characters=True)
	graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
	graph.write_png('decisiontree.png')
	img = mpimg.imread('decisiontree.png')
	plt.imshow(img)
	plt.axis('off')
	plt.show()

print('Start building decisiontree graph...')
p = Process(target=plot_graph)
p.start()

#predicted = dtc.predict(X_test)
#print('Decision tree score: ' , score(predicted))

cv_scores = cross_val_score(dtc, X, y, cv=10)
print('\nCross validation score: ' , cv_scores.mean())

rfc = RandomForestClassifier(n_estimators=10)
rfc = rfc.fit(X_train, y_train.ravel())

predicted = rfc.predict(X_test)
print('Random forest score: ', score(predicted))

C = 1.0
svc = svm.SVC(kernel='linear', C=C, gamma='auto').fit(X_train, y_train.ravel())

predicted = svc.predict(X_test)
print('\nSVM (linear) score: ', score(predicted))

svc = svm.SVC(kernel='rbf', C=C, gamma='auto').fit(X_train, y_train.ravel())
predicted = svc.predict(X_test)
print('SVM (rbf) score: ', score(predicted))

svc = svm.SVC(kernel='sigmoid', C=C, gamma='auto').fit(X_train, y_train.ravel())
predicted = svc.predict(X_test)
print('SVM (sigmoid) score: ', score(predicted))

svc = svm.SVC(kernel='poly', C=C, gamma='auto').fit(X_train, y_train.ravel())
predicted = svc.predict(X_test)
print('SVM (poly) score: ', score(predicted))


neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(X_train, y_train.ravel()) 
predicted = neigh.predict(X_test)
print('\nKNeighbors scores:')
print('K=10 ->', score(predicted))
print('Finding best value of K in range 1 to 50...')
K = 0
best = 0
for i in range(1, 50):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, y_train.ravel()) 

    predicted = neigh.predict(X_test)
    now = score(predicted)
    if (now > best):
        K = i
        best = now
        print('K=' + str(K) + ' ->', best)


scaler = MinMaxScaler()
all_features_minmax = scaler.fit_transform(X)

clf = MultinomialNB()
cv_scores = cross_val_score(clf, all_features_minmax, y.ravel(), cv=10)
print('\nNaive Bayes score: ', cv_scores.mean())


clf = LogisticRegression(solver='lbfgs')
cv_scores = cross_val_score(clf, X, y.ravel(), cv=10)
print('\nLogistic regression score: ', cv_scores.mean())

def create_model():
	model = Sequential()
	model.add(Dense(6, activation='relu', input_dim=4))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

print('\nNeural network...')

create_model().summary()

#history = model.fit(X_train, y_train.ravel(), batch_size=50, epochs=100, verbose=0, validation_data=(X_test, y_test))
#score = model.evaluate(X_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

estimator = KerasClassifier(build_fn=create_model, epochs=100, verbose=0)
cv_scores = cross_val_score(estimator, X, y, cv=10)
print('Score: ', cv_scores.mean())

import matplotlib.pyplot as plt
import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
from sklearn.cross_validation import*
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

# import iris data
iris = datasets.load_iris()
#print(iris)
#{'feature_names': ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'], 'target'
X = iris.data[:, :2]
#print(X)
# we only take the first two features. We could
 # avoid this ugly slicing by using a two-dim dataset
y = iris.target
#do the dimensional reduction using PCA
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)
# Get the 80% data to be the main data to deal with
pca = PCA(n_components=0.8)
pca.fit(X_train)

#format data
X_t_train = pca.transform(X_train)
X_t_test = pca.transform(X_test)

#built the svm model
#find the best gama and cost
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
#built dictionary to make a pair gamma with cost
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X_t_train, y_train)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

#svcmodel = SVC(kernel='rbf', C=1,gamma=0.1).fit(X_t_train,y_train)
svcmodel = SVC(kernel='linear', C=1,gamma=0.1).fit(X_t_train,y_train)
#print score of the model
print("score : ", svcmodel.score(X_t_test,y_test))
#print the predict result depend on trainning
print("predict target : ", svcmodel.predict(X_t_test))

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
 np.arange(y_min, y_max, h))

plt.subplot(1, 1, 1)
Z = svcmodel.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()
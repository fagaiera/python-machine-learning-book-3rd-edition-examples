from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from plotter import plot_decision_regions
import matplotlib.pyplot as plt
import numpy as np

# Loading the Iris dataset from scikit-learn. 
# Here, the third column represents the petal length, 
# and the fourth column the petal width of the flower examples. 
# The classes are already converted to integer labels 
# where 0=Iris-Setosa, 1=Iris-Versicolor, 2=Iris-Virginica.


iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# Splitting data into 70% training and 30% test data.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

forest = RandomForestClassifier(criterion='gini',
                                n_estimators=25, 
                                random_state=1,
                                n_jobs=2)
forest.fit(X_train, y_train)

plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105, 150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
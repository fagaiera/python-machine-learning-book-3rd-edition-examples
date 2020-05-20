from sklearn.svm import SVC
from sklearn import datasets
from plotter import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# Standardizing the features.

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
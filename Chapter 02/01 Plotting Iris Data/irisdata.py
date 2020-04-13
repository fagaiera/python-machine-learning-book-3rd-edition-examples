import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""Read the data frame"""
df = pd.read_csv('iris.data', header=None)
"""
Shows the tail of the data frame
print(df.tail())
"""

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

# plot data

fig = plt.figure()
fig.canvas.set_window_title('Iris Data')

plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')


plt.show()
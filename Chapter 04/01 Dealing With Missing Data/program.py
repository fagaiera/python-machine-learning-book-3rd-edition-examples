import pandas as pd
import numpy as np

from io import StringIO
from sklearn.impute import SimpleImputer


# Dealing with missing data

# Identifying missing values in tabular data

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
print(df)

print(df.isnull().sum())

# access the underlying NumPy array
# via the `values` attribute
print(df.values)

# Eliminating training examples or features with missing values

# remove rows that contain missing values

print(df.dropna(axis=0))

# remove columns that contain missing values

print(df.dropna(axis=1))

# remove columns that contain missing values

print(df.dropna(axis=1))

# only drop rows where all columns are NaN

print(df.dropna(how='all'))

# drop rows that have fewer than 3 real values 

print(df.dropna(thresh=4))

# only drop rows where NaN appear in specific columns (here: 'C')

print(df.dropna(subset=['C']))

# Imputing missing values

# again: our original array
print(df.values)

# impute missing values via the column mean
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
print(imputed_data)

print(df.fillna(df.mean()))
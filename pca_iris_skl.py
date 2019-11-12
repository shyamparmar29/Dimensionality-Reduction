'''
A Summary of the PCA Approach

1. Standardize the data.
2. Obtain the Eigenvectors and Eigenvalues from the covariance matrix by performing Singular Vector Decomposition.
3. Sort eigenvalues in descending order and choose the k eigenvectors that correspond to the k largest eigenvalues where k is the number of dimensions of the new feature subspace (k≤d)/.
4. Construct the projection matrix W from the selected k eigenvectors.
5. Transform the original dataset X via W to obtain a k-dimensional feature subspace Y.
'''

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])

features = ['sepal length', 'sepal width', 'petal length', 'petal width']

# Separating out the features
x = df.loc[:, features].values   #all rows and feature as columns
# Separating out the target
y = df.loc[:,['target']].values  #all rows and target as column

# Standardizing the features
X_std = StandardScaler().fit_transform(x)

#Covariance matrix
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)


#eigendecomposition
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

#SVD
#u,s,v = np.linalg.svd(X_std.T)
#print(u)

'''
The eigenvectors with the lowest eigenvalues bear the least information about the distribution of the data;
those are the ones can be dropped.
'''

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
print('\nEigenpairs \n%s' %eig_pairs)
# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

#explained variance
tot = sum(eig_vals)
exp_var = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
print(exp_var)


#Projection Matrix.

#Here, we are reducing the 4-dimensional feature space to a 2-dimensional feature subspace, by choosing 
#the "top 2" eigenvectors with the highest eigenvalues to construct our d×k-dimensional eigenvector matrix W.


matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1), 
                      eig_pairs[1][1].reshape(4,1)))

print('Matrix W:\n', matrix_w)

Y = X_std.dot(matrix_w) #Y is a 150×2 matrix of our transformed samples
print(Y)
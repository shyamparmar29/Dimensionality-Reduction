# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 23:32:40 2019

@author: Shyam Parmar
"""

import pandas as pd
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing   #Preprocessing for scaling data
import matplotlib.pyplot as plt
 
#########################
#
# Data Generation Code
#
#########################
## In this example, the data is in a data frame called data.
## Columns are individual samples (i.e. cells)
## Rows are measurements taken for all the samples (i.e. genes)
## Just for the sake of the example, we'll use made up data...
genes = ['gene' + str(i) for i in range(1,101)]
 
wt = ['wt' + str(i) for i in range(1,6)]
ko = ['ko' + str(i) for i in range(1,6)]
 
data = pd.DataFrame(columns=[*wt, *ko], index=genes)#The '*' unpacksthe wt and ko arrays so that the column 
#names are single arrays that look like [wt1, wt2,..wt6,ko1,ko2,..ko6]. Without the stars we'd create an array
#of two arrays and that would'nt create 12 columns like we want.[[w1,w2..w6],[ko1,ko2..ko6]]

#index=genes means gene names are used for the index which means they are the equivalent of row names.
 
for gene in data.index:
    data.loc[gene,'wt1':'wt5'] = np.random.poisson(lam=rd.randrange(10,1000), size=5)
    data.loc[gene,'ko1':'ko5'] = np.random.poisson(lam=rd.randrange(10,1000), size=5)
 
print(data.head())
print(data.shape)
 
#########################
#
# Perform PCA on the data
#
#########################
# First center and scale the data
# After centering, the avg val for each gene will be 0 and after scaling the standard deviation for the 
# values of each gene will be 1
scaled_data = preprocessing.scale(data.T)  #.T represents that we're passing transpose of our data. The scale
#function expects the samples to be rows instead of columns

#StandardScaler().fit_transform(data.T)  - This is more commonly used than processing.scale(data.T)
 
pca = PCA() # create a PCA object
pca.fit(scaled_data) # do the math
pca_data = pca.transform(scaled_data) # get PCA coordinates for scaled_data
 
#########################
#
# Draw a scree plot and a PCA plot
#
#########################
 
#The following code constructs the Scree plot
per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1) #Calculate %of variation that each PC accounts for
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]  # Labels for scree plot
 
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()
 
#the following code makes a fancy looking PCA plot using PC1 and PC2
pca_df = pd.DataFrame(pca_data, index=[*wt, *ko], columns=labels) #to draw PCA plot, we'll first put 
#thr new coordinates, created bt pca.transform(scaled.data), into a nice matrix where rows have sample lables
#and columns have pc labels
 
plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('My PCA Graph')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))
 
for sample in pca_df.index: #This loop adds sample names to the graph
    plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample])) 
 
plt.show()
 
#########################
#
# Determine which genes had the biggest influence on PC1
#
#########################
 
## get the name of the top 10 measurements (genes) that contribute
## most to pc1.
## first, get the loading scores
loading_scores = pd.Series(pca.components_[0], index=genes)
## now sort the loading scores based on their magnitude
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
 
# get the names of the top 10 genes
top_10_genes = sorted_loading_scores[0:10].index.values
 
## print the gene names and their scores (and +/- sign)
print(loading_scores[top_10_genes])
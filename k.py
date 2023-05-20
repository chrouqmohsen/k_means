#import libraries
import numpy as np
import os
import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import random as rd
# Load the data set
iris =load_iris()
X,Y = iris.data , iris.target

# Using  function to shuffle
def shuffle(X,Y):
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    return X[idx], Y[idx] 
iris.data,  iris.target = shuffle(iris.data,  iris.target)


# Set the number of clusters k
k = 3

# Select k random points from the data as centroids
centroid_indices = np.random.choice(X.shape[0], size=k, replace=False)
centroids = X[centroid_indices]
labels=[]
new_labels=[]
for i in range(len(centroids)):
    labels.append([])
    new_labels.append([])

# Assign all the points to the closest cluster centroid
distances = np.zeros((X.shape[0], k))
for i in range(k):
      distances = np.sum(abs(X[i]-centroids))
   #Returns the indices of the minimum values along an axis. 
#labels = np.argmin(distances, axis=1)

# Iterate until convergence
max_iter = 1000
for i in range(max_iter):
    
    distances = np.zeros((X.shape[0], k))
    for j in range(k):
        distances[:, j] = np.sqrt(np.sum((X - centroids[j])**2, axis=1))
        
    new_labels = np.argmin(distances, axis=1)
    
  #Check for convergence
    if np.all(labels == new_labels):
        break
    labels = new_labels

# Print the final cluster assignments
print(labels)
print("##################")
print(distances)
print("##################")
print(centroids)
print("##################")

print (X[0] ,X[2])
print("##################")
print(X[i])
print("##################")
print(iris.data)
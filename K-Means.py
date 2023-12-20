#######################################################################################
#K Means Algorithm #
#Authors: Jaime Nepomuceno Jiménez
#######################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from math import sqrt
import os

# Dataframe directory
directory = os.path.dirname(os.path.abspath(__file__))
archivo = directory + '\iris2.csv'

def assign_clusters(data, centroids):
    distances = np.zeros((length,2))
    assignments = np.zeros(length)
    for i in range(length):
        # Get euclidean distances
        distances[i,0] = sqrt((centroids[0,0] - data[i,0])**2 + (centroids[0,1] - data[i,1])**2 + 
                            (centroids[0,2] - data[i,2])**2 + (centroids[0,3] - data[i,3])**2)
        distances[i,1] = sqrt((centroids[1,0] - data[i,0])**2 + (centroids[1,1] - data[i,1])**2 + 
                            (centroids[1,2] - data[i,2])**2 + (centroids[1,3] - data[i,3])**2)
        # Assign clusters
        if distances[i,0] >= distances[i,1]:
            assignments[i] = 0
        else:
            assignments[i] = 1
    return assignments

def update_centroids(data, assignments, k):
    new_centroids = np.zeros((k, 4))
    for i in range(k):
        # if there is a value assigned to this cluster
        if np.sum(assignments == i) > 0:
            new_centroids[i] = np.mean(data[assignments == i], axis=0)
        # if no values is assigned
        else: 
            new_centroids[i] = data[np.random.choice(range(len(data)))]
    return new_centroids

# Read the dataset and get it´s values
df = pd.read_csv(archivo)
dataset = np.array(df.values)
length = np.size(dataset,0)

#Get max and min of each column
x1 = np.zeros(shape=np.size(dataset,0))
x2 = np.zeros(shape=np.size(dataset,0))
x3 = np.zeros(shape=np.size(dataset,0))
x4 = np.zeros(shape=np.size(dataset,0))

for i in range(length):
    x1[i] = dataset[i,0]
    x2[i] = dataset[i,1]
    x3[i] = dataset[i,2]
    x4[i] = dataset[i,3]
    
x1_max = np.amax(x1)
x1_min = np.amin(x1)
x2_max = np.amax(x2)
x2_min = np.amin(x2)
x3_max = np.amax(x3)
x3_min = np.amin(x3)
x4_max = np.amax(x4)
x4_min = np.amin(x4)

# Normalize each column with Minmax scaler
for i in range(length):
    x1[i] = (x1[i] - x1_min) / (x1_max - x1_min)
    x2[i] = (x2[i] - x2_min) / (x2_max - x2_min)
    x3[i] = (x3[i] - x3_min) / (x3_max - x3_min)
    x4[i] = (x4[i] - x4_min) / (x4_max - x4_min)

# Normalized dataframe
data = np.zeros((length,4))
for i in range(length):
    data[i,0] = x1[i]
    data[i,1] = x2[i]
    data[i,2] = x3[i]
    data[i,3] = x4[i]

#initial random  centroids
k = 2
centroids = np.zeros((k,4))
for i in range(k):
    centroids[i,0] = random.uniform(x1_min,x1_max)
    centroids[i,1] = random.uniform(x2_min,x2_max)
    centroids[i,2] = random.uniform(x3_min,x3_max)
    centroids[i,3] = random.uniform(x4_min,x4_max)

max_iter = 100
# Main loop
while i <= max_iter:
    #assign clusters to each point
    assignments = assign_clusters(data, centroids)
    #save old centroids
    old_centroids = centroids.copy()
    #update centroids
    centroids = update_centroids(data, assignments, k)
    i = i + 1

# Best clusters
print("Best centroids:")
print("Centroid 1: " + str(centroids[0]))
print("Centroid 2: " + str(centroids[1]))

# clusters view using only 2 dimensions
colors = ['r', 'b']
for i in range(k):
    cluster_points = data[assignments == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Cluster {i + 1}')
plt.scatter(centroids[:, 0], centroids[:, 1], c='g', marker='X', label='Centroids')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('K-means')
plt.show()
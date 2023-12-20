#######################################################################################
#K Nearest Neighbors (K-NN) Algorithm #
#Authors: Jaime Nepomuceno Jiménez
#######################################################################################

import numpy as np
import pandas as pd
import random
import csv
from math import sqrt
import os

# Shuffle the values of the dataframe
directory = os.path.dirname(os.path.abspath(__file__))
archivo_original = directory + '\iris.csv'
archivo_mezclado = directory + '\iris_shuffle.csv'
if not os.path.exists(archivo_mezclado):
    with open(archivo_original, 'r') as archivo_csv:
        lector_csv = csv.reader(archivo_csv)
        # save the header
        encabezados = next(lector_csv)  
        # save each row
        filas = list(lector_csv) 

    # shuffle the rows
    random.shuffle(filas)

    # write the rows in a new csv file
    with open(archivo_mezclado, 'w', newline='') as archivo_mezclado:
        escritor_csv = csv.writer(archivo_mezclado)
        escritor_csv.writerow(encabezados)  
        escritor_csv.writerows(filas)  

def calc_distances(rand_point, data, length):
    # initialize the distances array
    distancias = np.zeros((length,2))
    # loop to get the distances
    for i in range(length):
        distancias[i,0] = sqrt((rand_point[0] - data[i,0])**2 + 
                                (rand_point[1] - data[i,1])**2 + 
                                (rand_point[2] - data[i,2])**2 + 
                                (rand_point[3] - data[i,3])**2)
        if data[i,4] == 'Setosa' or data[i,4] == 0:
            distancias[i,1] = 0
        elif data[i,4] == 'Versicolor' or data[i,4] == 1:
            distancias[i,1] = 1
        elif data[i,4] == 'Virginica' or data[i,4] == 2:
            distancias[i,1] = 2
    return distancias

def knn(test_point, data, k):
    #reset values
    c0 = 0
    c1 = 0
    c2 = 0
    #Get euclidean distances of each point
    distancias = calc_distances(test_point,data, np.size((data),0))
    #Sort the distances
    ordenada = sorted(distancias, key=lambda x: x[0])
    #Get the k nearest distances
    for j in range(k):
        nearest[j] = ordenada[j] 
    # count the ocurrence of classes
    for j in range(k):
        if nearest[j,1] == 0:
            c0 = c0 + 1
        elif nearest[j,1] == 1:
            c1 = c1 + 1
        elif nearest[j,1] == 2:
            c2 = c2 + 1
    # Set new class predicted
    if c0 >= c1 and c0 >= c2:
        predicted_class = 0
    elif c1 >= c0 and c1 >= c2:
        predicted_class = 1
    else:
        predicted_class = 2
    return predicted_class

# Read the dataset and get it´s values
df = pd.read_csv("c:/Users/jaime/Desktop/SBC/iris_shuffle.csv")
dataset = np.array(df.values)
length = np.size(dataset,0)
#Convert target class to inegrer
for i in range(length):
    if dataset[i,4] == 'Setosa' or dataset[i,4] == 0:
        dataset[i,4] = 0
    elif dataset[i,4] == 'Versicolor' or dataset[i,4] == 1:
        dataset[i,4] = 1
    elif dataset[i,4] == 'Virginica' or dataset[i,4] == 2:
        dataset[i,4] = 2
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

# split the dataframe in 80% (train data) - 20% (test data) 
data_80 = np.zeros((int(length * .8),5))
data_20 = np.zeros((int(length * .2),5))
for i in range(length):
	if i < (length * .8):
		data_80[i,0] = x1[i]
		data_80[i,1] = x2[i]
		data_80[i,2] = x3[i]
		data_80[i,3] = x4[i]
		data_80[i,4] = dataset[i,4]
	elif i >= (length * .8):
		data_20[int(i - (length * .8)),0] = x1[i]
		data_20[int(i - (length * .8)),1] = x2[i]
		data_20[int(i - (length * .8)),2] = x3[i]
		data_20[int(i - (length * .8)),3] = x4[i]
		data_20[int(i - (length * .8)),4] = dataset[i,4]

# Define variables
k_range_max = 10
best_rate = 0
best_k = 0
nearest = np.zeros((k_range_max,2))

# loop to get best k value
for k in range(1, k_range_max + 1):
    success_rate = 0
    for i in range(np.size((data_20),0)):
        predicted_class = knn(data_20[i], data_80, k)
        if predicted_class == data_20[i,4]:
            success_rate = success_rate + 1
    success_rate = success_rate / np.size(data_20,0)
    # print("success rate: " + str(success_rate) + ", k value: " + str(k))
    if success_rate > best_rate:
        best_k = k
        best_rate = success_rate

success_rate = 0
# Show the classification results with the best k
print("original - predicted")
for i in range(np.size((data_20),0)):
    same_class = " "
    predicted_class = knn(data_20[i], data_80, best_k)
    if predicted_class == data_20[i,4]:
        success_rate = success_rate + 1
        same_class = " classified"
    print(str(int(data_20[i,4])) + " - " + str(predicted_class) + same_class)

# Final output
print("same class: " + str(success_rate) + " of: " + str(np.size((data_20),0)))
print("Classification rate: " + str(best_rate * 100) + "%")
print("Best k: " + str(best_k))
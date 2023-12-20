#######################################################################################
#Gradient Descent (GD) Algorithm #
#Authors: # Jaime Nepomuceno Jimnénez
#Maximum Likelihood Estimation (MLE) #
#Authors: Jaime Nepomuceno Jiménez
#######################################################################################

# Import required libraries.
import numpy as np
import matplotlib.pyplot as plt

#-------Initial Data-------#

x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
y = [0, 1, 1, 1]
data = np.zeros((4,3))
n = len(x1)
for i in range(n):
    data[i,0] = x1[i]
    data[i,1] = x2[i]
    data[i,2] = y[i]

def G(theta, i):
    return  1 / (1 + np.e**-(theta[0,2] * 1 + theta[0,0] * x1[i] + theta[0,1] * x2[i]))

def MLE(theta):
    sum = 0
    for i in  range(n):
        g = np.log(G(theta, i))
        g1 = np.log(1 - G(theta, i))

        # if g >= 0.5:
        #     g = 1
        # else:
        #     g = 0
        # if g1 >= 0.5:
        #     g1 = 1
        # else:
        #     g1 = 0

        sum = sum + (y[i] * g + (1 - y[i]) * g1)
    return (-1 * (sum/n))

# The maximum number of iterations is established.
dimensions = 3

# The limits of the search space are defined.
t = np.array([-5, 5])
f_range = np.tile(t, (dimensions, 1))

# The maximum number of iterations is established.
max_iter = 1000

# The population size is defined, as well as the variable
# to hold the population elements.
num_agents = 1
agents = np.zeros((num_agents, dimensions))

# Initialization process for the initial population. In this example code,
# agents is the variable which represents the population of num_agents.
for i in range(dimensions):
    dim_f_range = f_range[i, 1] - f_range[i, 0]
    agents[:, i] = np.random.rand(num_agents) * dim_f_range + f_range[i, 0]

best_position = np.zeros(dimensions)
best_fitness = np.nan
fitness = np.empty(num_agents)

# The best solution and the best fitness value for the initial population is obtained.
for i in range(num_agents):
    fitness[i] = MLE(np.array([agents[i]]))
    if i == 0:
        best_position = agents[i]
        best_fitness = fitness[i]
    elif fitness[i] < best_fitness:
        best_position = agents[i]
        best_fitness = fitness[i]

initialbest = best_fitness
iterbest = 0
initialPop = agents.copy()
initialFitness = fitness.copy()

# The iteration counter is defined.
iter = 0

aux_selector = np.arange(num_agents)

#alpha and delta values
alpha = 0.05
delta = 0.01

#initial point
# indexes = aux_selector
# indexes = np.random.choice(indexes, 1, replace=False)
# agents_selected = agents[indexes]
main_theta = agents  

#stop criteria variable
no_new_best = 0

#Main loop process for the optimization process.
#Here the entire functionality of the GD algorithm is implemented.
while iter < max_iter:
    #GD function  
    #convert theta + delta into a new array
    td0 = main_theta[0,0] + delta
    td1 = main_theta[0,1] + delta
    td2 = main_theta[0,2] + delta
    tdelta0 = np.array([td0, main_theta[0,1], main_theta[0,2]], ndmin=2)
    tdelta1 = np.array([main_theta[0,0], td1, main_theta[0,2]], ndmin=2)
    tdelta2 = np.array([main_theta[0,0], main_theta[0,1], td2], ndmin=2)
        
    #New  values of tetha
    main_theta[0,0] = main_theta[0,0] - alpha * (MLE(tdelta0) - MLE(main_theta)) / delta
    main_theta[0,1] = main_theta[0,1] - alpha * (MLE(tdelta1) - MLE(main_theta)) / delta
    main_theta[0,2] = main_theta[0,2] - alpha * (MLE(tdelta2) - MLE(main_theta)) / delta

    #compare new values with the limits
    for j in range(dimensions):
        upper_limit = f_range[j, 1]
        lower_limit = f_range[j, 0]
        #adjust the new values with the limits
        if main_theta[0,j] < lower_limit:
            main_theta[0,j] = lower_limit
        elif main_theta[0,j] > upper_limit:
            main_theta[0,j] = upper_limit

    #Get a new fitnes with the new data
    new_fitness = MLE(main_theta)

    # The replacement mechanism is then performed.
    if new_fitness < fitness:
        agents = main_theta
        fitness = new_fitness
        if fitness < best_fitness:
            best_position = agents
            best_fitness = fitness
            iterbest = iter + 1
            no_new_best = 0
            print(fitness)
            
    iter = iter + 1
    print("Iteration: " + str(iter))

    #stop criteria
    no_new_best = no_new_best + 1
    if no_new_best == 50 :
        iter = max_iter

#The best solution (decision variables) as well
#as the best fitness value for the optimization process is showed.
print("Best solution: " + str(best_position[0,0]) + ", " + str(best_position[0,1]) + ", " + str(best_position[0,2]))
print("Best fitness: " + str(best_fitness))
print("Initial fitness: " + str(initialbest))
print("Iteration where best is found: " + str(iterbest))

# a = input("ingresa 2 valores 0 o 1: ")
# b = input()


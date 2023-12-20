#######################################################################################
#Differential Evolution (DE) Algorithm #
#Authors: # Jorge Galvez, UdG
#Minimun Square Error function #
#Authors: Jaime Nepomuceno Jiménez
#######################################################################################

# Import required libraries.
import numpy as np
import matplotlib.pyplot as plt

def MSE(theta):
    #   Initial Data
    x = [1, 2, 3, 4]
    y = [1, 2, 3, 4]
    n = len(x)
    sum = 0
    #------------------------ MSE Formula --------------------------------#
    #-------------------Σ((yi - (mx + b))^2)/n----------------------------#
    for i in  range(n):
        sum = sum + (y[i] - ((theta[0,1] * 1) + (theta[0,0] * x[i])))**2
    return sum/n

# The maximum number of iterations is established.
dimensions = 2

# The limits of the search space are defined.
t = np.array([-10, 10])
f_range = np.tile(t, (dimensions, 1))

# The maximum number of iterations is established.
max_iter = 100

# The population size is defined, as well as the variable
# to hold the population elements.
num_agents = 15
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
    fitness[i] = MSE(np.array([agents[i]]))
    if i == 0:
        best_position = agents[i]
        best_fitness = fitness[i]
    elif fitness[i] < best_fitness:
        best_position = agents[i]
        best_fitness = fitness[i]

initialPop = agents.copy()
initialFitness = fitness.copy()

# The iteration counter is defined.
iter = 0

aux_selector = np.arange(num_agents)

# The scaling factor of the algorithm is established.
m = 0.5

# The cross factor of the algorithm is established.
cross_p = 0.2


# Main loop process for the optimization process.
# Here the entire functionality of the DE algorithm is implemented.
while iter < max_iter:
    for i in range(agents.shape[0]):
        # Three different individuals are chosen.
        indexes = aux_selector[aux_selector != i]
        indexes = np.random.choice(indexes, 3, replace=False)
        agents_selected = agents[indexes]
        # The crossover operation  operation is performed to obtain the mutant vector.
        mut = agents_selected[0] + m * (agents_selected[1] - agents_selected[2])
        # The differential mutation  of the DE algorithm is performed.
        prob_vector = np.random.rand(dimensions) <= cross_p
        mut = agents[i] * prob_vector + mut * np.logical_not(prob_vector)
    
        # It is verified that the generated vector is
        # within the search space defined by the upper and lower limits.
        for j in range(dimensions):
            upper_limit = f_range[j, 1]
            lower_limit = f_range[j, 0]

            if mut[j] < lower_limit:
                mut[j] = lower_limit
            elif mut[j] > upper_limit:
                mut[j] = upper_limit

        # The fitness value of the mutant vector is obtained.
        fitness_mut = MSE(np.array([mut]))

        # The replacement mechanism is then performed.
        if fitness_mut < fitness[i]:
            agents[i] = mut
            fitness[i] = fitness_mut
            if fitness[i] < best_fitness:
                best_position = agents[i]
                best_fitness = fitness[i]

        iter = iter + 1
        print("Iteration: " + str(iter))

# The best solution (decision variables) as well
# as the best fitness value for the optimization process is showed.
print("Best solution: " + "m: " + str(best_position[0]) + ", bias: " + str(best_position[1]))
print("Best fitness: " + str(best_fitness))

#Function Graphs
xGraph = np.linspace(-10, 10, 25)
yGraph = np.linspace(-10, 10, 25)
xv, yv = np.meshgrid(xGraph, yGraph)
fitnessGraph = np.zeros((25, 25))
for i in range(25):
    for j in range(25):
        arr = [[xv[i, j], yv[i, j]]]
        fitnessGraph[i, j] = MSE(np.asarray(arr))
plt.ion()
fig = plt.figure(1)
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.title('Minimum Square Error Function', fontsize=20)
ax.plot_surface(xv, yv, fitnessGraph, alpha=0.6)
ax.scatter(initialPop[:, 0], initialPop[:, 1], initialFitness[:], c='purple', s=10, marker="x")
ax.scatter(agents[:, 0], agents[:, 1], fitness[:], c='red', s=10, marker="x")
plt.show(block = True)

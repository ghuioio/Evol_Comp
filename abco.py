import numpy as np
import random
import math
import os

def target_func(var = [0,0]):
    return var[0]**2 + var[1]**2

def init_sources(food_sources = 3, min_values = [0,0], max_values= [6,6], target_func=target_func):
    sources = np.zeros((food_sources, len(min_values)+1)) # source có 3 cột bao gồm p1, p2, f(p1,p2) 
    for i in range(0,food_sources):
        for j in range(0, len(min_values)):
            sources[i, j] = random.uniform(min_values[j], max_values[j])
        sources[i, -1] = target_func(sources[i, 0:sources.shape[1] - 1])    
    return sources
# print(init_sources())

def fit_calc(func_values):
    if(func_values >=0):
        return 1.0/ (1.0 + func_values)
    else:
        return 1.0 - func_values


def fit_func(searching_in_sources):
    fitness = np.zeros((searching_in_sources.shape[0], 2)) # 2 cột gồm fit và prob
    for i in range(0, fitness.shape[0]):
        fitness[i,0] = fit_calc(searching_in_sources[i, -1])
    fit_sum = fitness[:, 0].sum()
    fitness[0,1] = fitness[0,0]
    for i in range(1, fitness.shape[0]):
        fitness[i, 1] = (fitness[i,0] + fitness[i-1, 1])
    for i in range(0, fitness.shape[0]):
        fitness[i, 1] = fitness[i, 1]/fit_sum
    return fitness


#print(fit_func(init_sources()))

def roulette_wheel(fitness):
    ix = 0
    random =  np.random.rand(1)
    for i in range(0, fitness.shape[0]):
        if(random <= fitness[i, 1]):
            ix = i
            break
    return ix


def employed_bee(sources, min_values = [0,0], max_values = [6,6], target_func = target_func):
    searching_in_sources = np.copy(sources)
    new_solution = np.zeros((1, len(min_values)))
    trial = np.zeros((sources.shape[0], 1))
    phi = random.uniform(-1, 1)
    for i in range(0, searching_in_sources.shape[0]):
        # phi = random.uniform(-1, 1)
        j = np.random.randint(len(min_values), size=1)[0]
        k = np.random.randint(searching_in_sources.shape[0], size=1)[0]
        while i==k:
            k = np.random.randint(searching_in_sources.shape[0], size = 1)[0]
        xij = searching_in_sources[i, j]
        xkj = searching_in_sources[k, j]
        vij = xij + phi*(xij - xkj)
        for variable in range(0, len(min_values)):
            new_solution[0, variable] = searching_in_sources[i , variable]
        new_solution[0, j] = np.clip(vij, min_values[j], max_values[j])
        new_function_value = target_func(new_solution[0, 0:new_solution.shape[1]])
        if (fit_calc(new_function_value) > fit_calc(searching_in_sources[i,-1])):
            searching_in_sources[i,j] = new_solution[0,j]
            searching_in_sources[i, -1]= new_function_value
        else:
            trial[i,0] = trial[i,0] + 1
            phi = random.uniform(-1, 1) 
        for variable in range(0, len(min_values)):
            new_solution[0, variable] = 0.0
    return searching_in_sources, trial

# x, y = employed_bee(init_sources())
# print(x)

def outlooker_bee(searching_in_sources, fitness, trial, min_values = [0,0], max_values = [6,6], target_func = target_func):
    improving_sources = np.copy(searching_in_sources)
    new_solution      = np.zeros((1, len(min_values)))
    trial_update      = np.copy(trial)
    for repeat in range(0, improving_sources.shape[0]):
        i   = roulette_wheel(fitness)
        phi = random.uniform(-1, 1)
        j   = np.random.randint(len(min_values), size = 1)[0]
        k   = np.random.randint(improving_sources.shape[0], size = 1)[0]
        while i == k:
            k = np.random.randint(improving_sources.shape[0], size = 1)[0]
        xij = improving_sources[i, j]
        xkj = improving_sources[k, j]
        vij = xij + phi*(xij - xkj)      
        for variable in range(0, len(min_values)):
            new_solution[0, variable] = improving_sources[i, variable]
        new_solution[0, j] = np.clip(vij,  min_values[j], max_values[j])
        new_function_value = target_func(new_solution[0,0:new_solution.shape[1]])    
        if (fit_calc(new_function_value) > fit_calc(improving_sources[i,-1])):
            improving_sources[i,j]  = new_solution[0, j]
            improving_sources[i,-1] = new_function_value
            trial_update[i,0]       = 0
        else:
            trial_update[i,0] = trial_update[i,0] + 1      
        for variable in range(0, len(min_values)):
            new_solution[0, variable] = 0.0    
    return improving_sources, trial_update

def scouter_bee(improving_sources, trial_update, limit = 3, target_func = target_func):
    for i in range(0, improving_sources.shape[0]):
        if (trial_update[i,0] > limit):
            for j in range(0, improving_sources.shape[1] - 1):
                improving_sources[i,j] = np.random.normal(0, 1, 1)[0]
            function_value = target_func(improving_sources[i,0:improving_sources.shape[1]-1])
            improving_sources[i,-1] = function_value
    return improving_sources

def artificial_bee_colony_optimization(food_sources = 30, iterations = 10, min_values = [0,0], max_values = [6,6], employed_bees = 10, outlookers_bees = 10, limit = 20, target_func = target_func):  
    count = 0
    best_value = float("inf")
    sources = init_sources(food_sources = food_sources, min_values = min_values, max_values = max_values, target_func = target_func)
    fitness = fit_func(sources)
    while (count <= iterations):
        # if (count > 0):
        #     print("Iteration = ", count, " f(x) = ", best_value)
        #     print(best_solution)    
        e_bee = employed_bee(sources, min_values = min_values, max_values = max_values, target_func = target_func)
        for i in range(0, employed_bees - 1):
            e_bee = employed_bee(e_bee[0], min_values = min_values, max_values = max_values, target_func = target_func)
        fitness = fit_func(e_bee[0])          
        o_bee = outlooker_bee(e_bee[0], fitness, e_bee[1], min_values = min_values, max_values = max_values, target_func = target_func)
        for i in range(0, outlookers_bees - 1):
            o_bee = outlooker_bee(o_bee[0], fitness, o_bee[1], min_values = min_values, max_values = max_values, target_func = target_func)
        value = np.copy(o_bee[0][o_bee[0][:,-1].argsort()][0,:])
        if (best_value > value[-1]):
            best_solution = np.copy(value)
            best_value    = np.copy(value[-1])       
        sources = scouter_bee(o_bee[0], o_bee[1], limit = limit, target_func = target_func)  
        fitness = fit_func(sources)
        count = count + 1   
    print(best_solution) 
    return best_solution


def f(var = [0,0,0,0,0]):
    sum = 0
    for i in range(5):
        sum += var[i]**2
    return sum
# abc = artificial_bee_colony_optimization(food_sources = 3, iterations = 1000, min_values = [-100,-100,-100,-100,-100], max_values = [100,100,100,100,100], employed_bees = 10, outlookers_bees = 10, limit = 30, target_func= f)
# abc = artificial_bee_colony_optimization(food_sources = 3, iterations = 100, min_values = [-5,-5,-5,-5,-5], max_values = [5,5,5,5,5], employed_bees = 1, outlookers_bees = 1, limit = 3, target_func= f)




        

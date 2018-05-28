#!/usr/bin/python

import numpy as np

def SSA(update_matrix, initial_conditions, function_rate, t_max):
    
    def choose_reaction(rate, r):
        cum_rate = np.cumsum(rate)
        i = 0
	val = 0
        while cum_rate[i] < r:
            val = i + 1
            i += 1
        return val

    i = 0
    data_vector = []
    data_vector.append(initial_conditions)
    time_vector = []
    time_vector.append(0) #default
    app_T = 0

    while app_T < t_max:
        rate = function_rate(initial_conditions)
        total_rate = sum(rate)
        r1 = np.random.uniform(0.0001, total_rate * 0.99999) #to chose which reaction
        r2 = np.random.uniform(0.0001, 1)   #to chose how much time between reactions
        T = -1.0 / total_rate * np.log(r2)
	app_T = T + time_vector[i]
        time_vector.append(app_T)
        reaction = choose_reaction(rate, r1)
        initial_conditions = initial_conditions + update_matrix[reaction, :]
	data_vector.append(initial_conditions)
        i += 1
    
    return np.array(data_vector), time_vector

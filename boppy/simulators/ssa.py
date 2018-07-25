#!/usr/bin/python

from copy import deepcopy
import numpy as np


def _initialize_vector_binary_search(vector):
    """Generate a new vector of lenght (2 * m - 1), where m is the lenght of the input vector and
    the element in position i is the sum of elements in position (2 * i) and (2 * i + 1).
    """
    binary_vector = np.zeros(2 * vector.shape[0] - 1)  # total nodes for vector.shape[0] values
    binary_vector[vector.shape[0] - 1:] = vector

    i = vector.shape[0] - 2
    while i > -1:
        binary_vector[i] = binary_vector[2 * i + 1] + binary_vector[2 * i + 2]
        i -= 1
    return binary_vector


def _binary_search_processing(vector, random_value):
    """Perform binary search on a vector."""
    i = 0
    cut_value = (vector.shape[0] + 1) / 2 - 1
    while i < cut_value:
        if vector[2 * i + 1] >= random_value:
            i = 2 * i + 1
        else:
            random_value -= vector[2 * i + 1]
            i = 2 * i + 2

    return int(i - cut_value)


def SSA(update_matrix, initial_conditions, function_rate, t_max, **kwargs):
    """Stochastic Simulation Algorithm."""
    # vector of number of reaction for each times
    data_vector = [deepcopy(initial_conditions)]
    time_vector = [0]
    previous_states = deepcopy(initial_conditions)

    i = simul_t = 0
    while simul_t < t_max:
        rates = function_rate(previous_states)
        total_rate = sum(rates)

        # Generate two random numbers: first to choose reaction, second to choose execution time.
        rnd_react = np.random.uniform(0.0001, total_rate)
        rnd_time = np.random.uniform(0.0001, 1)

        simul_t = - np.log(rnd_time) / total_rate + time_vector[i]

        # choose reaction and update the vector of reactions
        vector_binary = _initialize_vector_binary_search(rates)
        reaction = _binary_search_processing(vector_binary, rnd_react)

        previous_states += update_matrix[reaction, :]

        # update the data_vector and time_vector
        time_vector.append(simul_t)
        data_vector.append(deepcopy(previous_states))

        i += 1

    return np.array(data_vector), np.array(time_vector)

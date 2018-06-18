#!/usr/bin/python

import numpy as np


def SSA(update_matrix, initial_conditions, function_rate, t_max, **kwargs):

    """Function that implement the Stochastic Simulation Algorithm"""
    
    def initialize_vector_binary_search(vector):
        """
        Function that generate a vector of lenght 2*m-1, where m is the lenght of the given vector,
        when the element at the position i is the sum of elements of position 2*i and 2*i + 1
                """
        binary_vector = np.zeros(shape = len(vector)-1) #total nodes for len(vector) values
        binary_vector = np.append(binary_vector, vector)
        i = len(vector)-2
        while i > -1:
            binary_vector[i] = binary_vector[2*i + 1] + binary_vector[2*i + 2]
            i -= 1
        return binary_vector

    def binary_search_processing(vector, random_value):
        """
        Function is used with a vector for binary search. Gives a component of last m component
        """
        i = 0
        cut_value = (len(vector)+1)/2 - 1
        verify = True
        while verify == True:
            if vector[2*i + 1] >= random_value:
                i = 2*i + 1
            else:
                random_value -= vector[2*i + 1]
                i = 2*i + 2
            if i >= cut_value:
                verify = False
    
        return int(i - cut_value)

    #vector of number of reaction for each times
    data_vector = []
    data_vector.append(initial_conditions)
    #vector of times
    time_vector = []
    time_vector.append(0)  # default
    app_t = 0

    while app_t < t_max:
        rate = function_rate(initial_conditions)
        total_rate = sum(rate)

        #Generate the two random number of algorithm: first to choose reaction, second to choose time 

        r1 = np.random.uniform(0.0001, total_rate) 
        r2 = np.random.uniform(0.0001, 1) 

        t = -1.0 / total_rate * np.log(r2)
        app_t = t + time_vector[i]

        #choose reaction and update the vector of reactions

        vector_binary = initialize_vector_binary_search(rate)
        reaction = binary_search_processing(vector_binary,r1)

        initial_conditions = initial_conditions + update_matrix[reaction, :]

        #update the data_vector and time_vector

        time_vector.append(app_t)
        data_vector.append(initial_conditions)

    return np.array(data_vector), time_vector

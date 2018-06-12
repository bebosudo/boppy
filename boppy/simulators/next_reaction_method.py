import numpy as np

np.seterr(divide='ignore', invalid='ignore')


def next_reaction_method(update_matrix, initial_mol_number, propensity_function, t_max):
    """
    This function implements the Next Reaction Method from Gibson and Bruck.

    References:
    M.A. Gibson and J.Bruck "Efficient Exact Stochastic Simulation of Chemical Systems with Many Species and Many Channels",
    The Journal of Physical Chemistry A, 2000, 104 (9), 1876-1889
    """

    # Initialize
    mol_number = np.copy(initial_mol_number)
    time_simul = 0

    num_reactions = np.shape(update_matrix)[0]

    trajectory_states = []   # States
    trajectory_ftimes = []   # Firing times

    # Calculate the propensity function for each reaction
    propensity_val = propensity_function(initial_mol_number)

    # Generate a putative time, according to an exponential distribution
    putative_times = -1 / propensity_val * \
        np.log(np.random.random(num_reactions))
    putative_times[np.isnan(putative_times)] = np.inf   # 0/0 set to inf

    while time_simul < t_max:

        trajectory_states.append(np.copy(mol_number))
        trajectory_ftimes.append(time_simul)

        # Select the reaction whose putative time is least
        next_reaction_index = np.argmin(putative_times)

        # Change the number of molecules to reflect execution of reaction
        mol_number += update_matrix[next_reaction_index]
        time_simul = putative_times[next_reaction_index]

        # Calculate the propensity functions after execution of reaction
        propensity_val_new = propensity_function(mol_number)

        # Update the putative times
        for reaction_index in range(0, num_reactions):
            if (reaction_index == next_reaction_index) or (propensity_val[reaction_index] == 0 and propensity_val_new[reaction_index] != 0):
                putative_times[reaction_index] = - 1 / propensity_val_new[reaction_index] * \
                    np.log(np.random.random(1)) + time_simul
            else:
                putative_times[reaction_index] = propensity_val[reaction_index] / propensity_val_new[reaction_index] * \
                    (putative_times[reaction_index] - time_simul) + time_simul

        putative_times[np.isnan(putative_times)] = np.inf

        # Update the propensity functions
        propensity_val = np.copy(propensity_val_new)

    return np.array(trajectory_states), np.array(trajectory_ftimes)

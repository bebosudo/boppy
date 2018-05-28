import numpy as np
from collections import defaultdict


def next_reaction_method(update_matrix, initial_mol_number, propensity_functions, affects, depends_on):
    """ 
    This function implements the Next Reaction Method from Gibson and Bruck.

    References:
    M.A. Gibson and J.Bruck "Efficient Exact Stochastic Simulation of Chemical Systems with Many Species and Many Channels",
    The Journal of Physical Chemistry A, 2000, 104 (9), 1876-1889
    """

    # Initialize
    mol_number = np.copy(initial_mol_number)
    time = 0
    n_reactions = np.shape(update_matrix)[0]

    # Generate a dependecy graph
    dependency_graph = defaultdict(set)
    for i in range(0, n_reactions):
        for j in range(0, n_reactions):
            if len(np.intersect1d(affects[i], depends_on[j])) != 0:
                dependency_graph[i].add(j)

    trajectory_mol_number = []   # Output
    trajectory_times = []

    # Calculate the propensity function for each reaction
    propensity_fun = propensity_functions(mol_number)

    # Generate a putative time, according to an exponential distribution, for
    # each reaction
    uniform_variables = np.random.random(n_reactions)
    putative_times = -1 / propensity_fun * np.log(uniform_variables)

    # Store putative times in an indexed priority queue
    times_ipq = np.copy(putative_times)
    reactions_ipq = [index for index in range(0, n_reactions)]
    built_ipq(times_ipq, reactions_ipq)

    while time < t_max:

        trajectory_mol_number.append(np.copy(mol_number))
        trajectory_times.append(time)

        # Select the reaction whose putative time is least
        time = times_ipq[0]
        next_reaction = reactions_ipq[0]

        # Change the number of molecules to reflect execution of reaction
        mol_number += update_matrix[next_reaction, :]

        # Calculate the propensity functions after execution of reaction
        propensity_fun_new = propensity_functions(mol_number)

        # Update the putative times
        for index_reaction in dependency_graph[next_reaction]:
            if index_reaction != next_reaction:
                putative_times[index_reaction] = propensity_fun[index_reaction] / propensity_fun_new[index_reaction] * \
                    (putative_times[index_reaction] - time) + \
                    time if putative_times[
                        index_reaction] != np.inf else np.inf
            else:
                putative_times[index_reaction] = time - 1 / \
                    propensity_fun_new[index_reaction] * \
                    np.log(np.random.random(1))

            # Update the indexed priority queue with the new value
            update_ipq(times_ipq, reactions_ipq, reactions_ipq.index(
                index_reaction), putative_times[index_reaction])

        # Update the propensity functions
        propensity_fun = np.copy(propensity_fun_new)

    return np.array(trajectory_mol_number), np.array(trajectory_times)


################################# INDEXED PRIORITY QUEUE #################

def swap_ipq(tree, inds, node_i, node_j):
    """ 
    Swaps the tree nodes node_i and node_j and updates the index structure inds appropriately
    """
    tree[node_i], tree[node_j] = tree[node_j], tree[node_i]
    inds[node_i], inds[node_j] = inds[node_j], inds[node_i]


def built_ipq(tree, inds):
    """
    Takes a tree and an index structure and moves entries untile the tree has the property
    that each parent is less than its children
    """
    for node in range(len(tree) // 2 - 1, -1, -1):
        heapify(tree, inds, node)


def heapify(tree, inds, node):
    smallest = node
    left = 2 * node + 1
    right = 2 * node + 2

    if left < len(tree) and tree[left] < tree[smallest]:
        smallest = left

    if right < len(tree) and tree[right] < tree[smallest]:
        smallest = right

    if smallest != node:
        swap_ipq(tree, inds, smallest, node)
        heapify(tree, inds, smallest)


def update_ipq(tree, inds, node, value):
    """ 
    Updates the indexed priority queue
    """
    tree[node] = value
    update_aux(tree, inds, node)


def update_aux(tree, inds, node):
    parent = (node - 1) // 2
    left = 2 * node + 1
    right = 2 * node + 2

    if left < len(tree):
        min_child = left
    if right < len(tree) and tree[min_child] > tree[right]:
        min_child = right

    if parent > 0 and tree[node] < tree[parent]:
        swap_ipq(tree, inds, node, parent)
        update_aux(tree, inds, parent)
    elif left < len(tree) and tree[node] > tree[min_child]:
        swap_ipq(tree, inds, node, min_child)
        update_aux(tree, inds, min_child)
    else:
        return

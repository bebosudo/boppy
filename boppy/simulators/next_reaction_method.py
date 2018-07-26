import numpy as np
import boppy.core
from collections import namedtuple
np.seterr(divide='ignore', invalid='ignore')


def next_reaction_method(update_matrix, initial_mol_number, propensity_function, t_max, **kwargs):
    """
    This function implements the Next Reaction Method from Gibson and Bruck.

    References:
    M.A. Gibson and J.Bruck "Efficient Exact Stochastic Simulation of Chemical Systems with Many Species and Many Channels",
    The Journal of Physical Chemistry A, 2000, 104 (9), 1876-1889
    """

    # Retrieve the vectors to use to build the dependency graph.
    depends_on_vector = kwargs['depends_on']
    affects_vector = kwargs['affects']

    # Initialize
    mol_number = np.copy(initial_mol_number)
    time_simul = 0

    num_reactions = np.shape(update_matrix)[0]

    trajectory_states = []   # States
    trajectory_ftimes = []   # Firing times

    # Generate a dependecy graph
    dependecy_graph = boppy.core.DependencyGraph(affects_vector, depends_on_vector)

    # Calculate the propensity function for each reaction
    propensity_val = propensity_function(initial_mol_number)

    # Generate putative times, according to an exponential distribution
    putative_times = -1 / propensity_val * \
        np.log(np.random.random(num_reactions))
    putative_times[np.isnan(putative_times)] = np.inf   # 0/0 set to inf

    # Store the putative times in a indexed priority queue
    nodes_list = [IPQnode(index, time) for index, time in enumerate(putative_times)]
    ipq = IndexedPriorityQueue(nodes_list)

    while time_simul < t_max:

        trajectory_states.append(np.copy(mol_number))
        trajectory_ftimes.append(time_simul)

        # Select the reaction whose putative time is least
        next_reaction_index = ipq.tree[0].index

        # Change the number of molecules to reflect execution of reaction
        mol_number += update_matrix[next_reaction_index]
        time_simul = putative_times[next_reaction_index]

        # Calculate the propensity functions after execution of reaction
        propensity_val_new = propensity_function(mol_number)

        # Update the putative times and the indexed priority queue
        for reaction_index in dependecy_graph.graph[next_reaction_index]:
            if (reaction_index == next_reaction_index) or (propensity_val[reaction_index] == 0 and propensity_val_new[reaction_index] != 0):
                putative_times[reaction_index] = - 1 / propensity_val_new[reaction_index] * \
                    np.log(np.random.random(1)) + time_simul
            else:
                putative_times[reaction_index] = propensity_val[reaction_index] / propensity_val_new[reaction_index] * \
                    (putative_times[reaction_index] - time_simul) + time_simul
            if np.isnan(putative_times[reaction_index]):
                putative_times[reaction_index] = np.inf

            ipq.update(IPQnode(reaction_index, putative_times[reaction_index]))

        # Update the propensity functions
        propensity_val = np.copy(propensity_val_new)

    # Pack together the time column with the states associated to it.
    return np.c_[trajectory_ftimes, trajectory_states]


IPQnode = namedtuple('Node', ['index', 'time'])  # Node of the Indexed Priority Queue


class IndexedPriorityQueue:
    """
    An indexed priority queue consists of a tree structure of ordered IPQnode (pairs of reaction's index and putative time)
    and a dictionary where the i-th key is mapped to the position in the tree that contains the IPQnode whose index is i.
    The tree structure has the property that each parent has a lower putative time than either of its children (heap).
    """

    def __init__(self, nodes_list):
        self.tree = nodes_list
        self.dict = {node.index: index for index, node in enumerate(nodes_list)}
        self._built()

    def _built(self):
        """Moves entries until the tree has the property that each parent is less than its children."""
        for node_index in range(len(self.tree) // 2 - 1, -1, -1):
            self._heapify(node_index)

    def _heapify(self, node_index):
        smallest = node_index
        left = 2 * node_index + 1
        right = 2 * node_index + 2

        if left < len(self.tree) and self.tree[left].time < self.tree[smallest].time:
            smallest = left

        if right < len(self.tree) and self.tree[right].time < self.tree[smallest].time:
            smallest = right

        if smallest != node_index:
            self._swap(smallest, node_index)
            self._heapify(smallest)

    def _swap(self, node_i, node_j):
        """Swaps the tree nodes node_i and node_j and updates the dictionary appropriately"""
        self.tree[node_i], self.tree[node_j] = self.tree[node_j], self.tree[node_i]
        self.dict[self.tree[node_i].index], self.dict[self.tree[node_j].index] = node_i, node_j

    def update(self, ipq_node):
        """Updates the indexed priority queue"""
        self.tree[self.dict[ipq_node.index]] = ipq_node
        self._update_aux(self.dict[ipq_node.index])

    def _update_aux(self, node_index):
        parent = (node_index - 1) // 2
        left = 2 * node_index + 1
        right = 2 * node_index + 2

        if left < len(self.tree):
            min_child = left
        if right < len(self.tree) and self.tree[min_child].time > self.tree[right].time:
            min_child = right

        if parent >= 0 and self.tree[node_index].time < self.tree[parent].time:
            self._swap(node_index, parent)
            self._update_aux(parent)
        elif left < len(self.tree) and self.tree[node_index].time > self.tree[min_child].time:
            self._swap(node_index, min_child)
            self._update_aux(min_child)
        else:
            return

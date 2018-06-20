import numpy as np
from scipy.integrate import odeint
from sympy import oo
import sympy as sym
import boppy.core


def fluid_approximation(update_matrix, initial_conditions, function_rate, t_max, **kwargs):
    """
    Secundary arguments. In these rate functions variable system size is represented by N.
    Also needed the costant system size.
    """
    rate_funcs_N = kwargs["rate_functions_var_ss"]
    variables = kwargs["variables"]
    system_size = kwargs["system_size"]

    def variables_involved(rate_func):
        """
        Extract the vector of species involved in a certain rate function. List of symbols.
        """
        sym_rate_func = rate_func.sym_function
        args = list(sym_rate_func.args)
        species_inv = [i for i in args if i.is_Symbol == True]
        return species_inv

    # Dictionary to substitute individuals variables symbols with densities variables symbols.
    var_to_substitute = {}
    for var_obj in variables.values():
        var_to_substitute[var_obj.symbol] = sym.Symbol('d_' + var_obj.str_var)

    d_initial_conditions = [x / system_size for x in initial_conditions]

    # Dictionary to substitute densities symbols with their numerical values.
    densities = {}
    for i, val in enumerate(var_to_substitute.values()):
        densities[val] = d_initial_conditions[i]

    def scaling(rate_functions_vector, substitutor_dict):
        """
        This function scales the rate functions, normalizing individuals variables with densities
        variables, and calculate the limit f for each one, a function required to be locally
        Lipschitz continuos and bounded. This function is needed to calculate the limit vector
        field (limit of the drift).
        """
        N = sym.Symbol('N')
        f_functions_vector = []
        for ratefun in rate_functions_vector:
            n = len(variables_involved(ratefun))
            ratefun = ratefun.sym_function
            ratefun = ratefun.subs(substitutor_dict)
            ratefun_normalized = ratefun * (N**n)
            f_N = ratefun_normalized/N
            f = sym.limit(f_N, N, oo)
            f_functions_vector.append(f)

        return f_functions_vector

    t = np.linspace(0, t_max, 1000)

    f_funcs = scaling(rate_funcs_N, var_to_substitute)

    print(f_funcs)

    # To complete from here
    def mydot(matrix, vector):
        pass

    # mydot(update_matrix, f_funcs)
    # another function here...
    # odeint(another function..., densities , t)

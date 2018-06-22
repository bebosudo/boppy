import numpy as np
import sympy as sym
from scipy.integrate import odeint


def fluid_approximation(update_matrix, initial_conditions, function_rate, t_max, **kwargs):
    """
    Secondary arguments. In these rate functions variable system size is represented by N.
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
        species_inv = [sym_el for sym_el in sym_rate_func.args if sym_el.is_symbol]
        return species_inv

    # Dictionary to substitute individuals variables symbols with densities variables symbols.
    var_to_substitute = {}
    for var_obj in variables.values():
        var_to_substitute[var_obj.symbol] = sym.Symbol('d_' + var_obj.str_var)

    d_initial_conditions = [x / system_size for x in initial_conditions]

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
            ratefun = ratefun.sym_function.subs(substitutor_dict)
            ratefun_normalized = ratefun * (N**n)
            f_N = ratefun_normalized / N
            f = sym.limit(f_N, N, sym.oo)
            f_functions_vector.append(f)

        return f_functions_vector

    t = np.linspace(0, t_max, 1000)

    f_funcs = scaling(rate_funcs_N, var_to_substitute)

    def create_equations(np_matrix, symbolic_functions_vector):
        """Matrix product between a numerical matrix and a vector of symbolic functions.

        It creates a list of linear equations.
        """
        eqs_list = []
        for row in np_matrix:
            eqs_list.append(sum(f * elem for f, elem in zip(symbolic_functions_vector, row)))

        return eqs_list

    def ode_model(x, t):
        """Generate the correct input for 'odeint'.

        Odeint is a function that needs a vector of initial conditions and time.

        TODO: Check whether a generator could be returned instead of a list.
        """
        return [equation.evalf(subs=zip(var_to_substitute.values(), x))
                for equation in create_equations(update_matrix, f_funcs)]

    trajectories_states = odeint(ode_model, d_initial_conditions, t)
    trajectories_times = t

    return np.array(trajectories_states), np.array(trajectories_times)

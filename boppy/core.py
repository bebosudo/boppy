import logging
import numpy as np
import sympy as sym
import numbers
from .utils import parser, misc
from .simulators import ssa, next_reaction_method
# import pdb
# pdb.set_trace()


InputError = misc.BoppyInputError

_LOGGER = logging.getLogger(__name__)

ALGORITHMS_AVAIL = ("ssa", "gillespie", "nrm", "next reaction method", "gibson bruck",
                    "gibson-bruck", "ode", "tau-leaping")


class Variable:
    """A Variable (or species) represents a set of individuals in the population."""

    def __init__(self, str_symbol, var_index):
        _LOGGER.debug("Creating a new Variable object: %s", str_symbol)
        self._var_index = var_index
        self._str_var = str_symbol
        self._sym_symbol = sym.Symbol(str_symbol)

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    @property
    def pos(self):
        return self._var_index

    @property
    def symbol(self):
        return self._sym_symbol

    def __hash__(self):
        return hash(self._str_var)

    def __str__(self):
        return self.__class__.__name__ + "(" + repr(self._str_var) + ")"

    def __repr__(self):
        return str(self)


class Parameter:
    """A Parameter is a constant rate."""

    def __init__(self, str_symbol, param_value):
        _LOGGER.debug("Creating a new Parameter object: %s", str_symbol)
        self._str_symbol = str_symbol
        self._sym_symbol = sym.Symbol(str_symbol)
        self._param_value = param_value

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        elif isinstance(other, str):
            return self._str_symbol == other
        return NotImplemented

    @property
    def symbol(self):
        return self._sym_symbol

    @property
    def value(self):
        return self._param_value

    def __hash__(self):
        return hash(self._str_symbol)

    def __str__(self):
        return self.__class__.__name__ + "(" + repr(self._str_symbol) + ")"

    def __repr__(self):
        return str(self)


class RateFunction:
    """A Rate Function depends on the Species (Variables) and on Parameters."""

    def __init__(self, str_rate_function, variables_collection, parameters_collection):
        self._orig_rate_function = str_rate_function
        self._variables = variables_collection
        self._parameters = parameters_collection

        self._pp_rate_function = parser.parse_function(
            self._orig_rate_function)
        _LOGGER.debug("Parsed '%s' into: '%s'",
                      self._orig_rate_function, self._pp_rate_function)

        tokenized_objs = (misc.Token(elem) for elem in self._pp_rate_function)
        rpn_tokens = parser.shunting_yard(tokenized_objs)
        _LOGGER.debug("Converted '%s' to RPN sequence: '%s'",
                      self._orig_rate_function, rpn_tokens)

        function_with_params = parser.rpn_calculator(rpn_tokens)
        _LOGGER.debug("Converted RPN sequence '%s' to symbolic function: '%s'",
                      rpn_tokens, function_with_params)

        # Convert back each Parameter object to {sympy object: actual value}
        # and substitute it.
        param_symbol_to_val = {
            val.symbol: val.value for val in parameters_collection.values()}
        self.sym_function = function_with_params.subs(param_symbol_to_val)
        _LOGGER.debug(
            "Substituted Parameters with their value; function '%s':", self.sym_function)

        self.lambdified = sym.lambdify(tuple(var.symbol for var in variables_collection.values()),
                                       self.sym_function)

    def function(self, arg):
        return self.lambdified(*arg)

    def __str__(self):
        return self.__class__.__name__ + "(" + repr(self.sym_function) + ")"

    def __repr__(self):
        return str(self)

    def __call__(self, vector):
        return self.function(vector)


class Reaction:
    """A Reaction is a combination of Variable(s) that produces other Variable(s) as output."""

    def __init__(self, str_reaction, variables_collection):
        _LOGGER.debug("Creating a new Reaction object: %s", str_reaction)
        self._orig_reaction = str_reaction
        self._variables = variables_collection
        self._parse_reaction()
        self.update_vector = self._produce_update_vector()
        self._depends_on()
        self._affects()

    def _parse_reaction(self):
        self._pp_reaction = parser.parse_reaction(self._orig_reaction)
        _LOGGER.debug("Parsed string %s to Reaction object: %s",
                      self._orig_reaction,
                      self._pp_reaction)

    def _extract_variable_from_input_list(self, symbol):
        try:
            return self._variables[symbol]
        except KeyError:
            raise InputError("Unable to find reagent '{}' inside the list of variables "
                             "provided {}".format(symbol, self._variables))

    def _produce_update_vector(self):
        """Create the update vector from the parsed reaction and the list of Variable objects.

        It searches for a match for each reagent, raising an exception if one of the variables in
        the pyparsing object is not present inside the list of Variable objects.
        """
        update_vec = np.zeros(len(self._variables), dtype=float)

        # Each reagent used in the reaction decreases the quantity available, while the products
        # increase it.
        for elements, multiplier in (("reagents", -1), ("products", 1)):
            for reagent in self._pp_reaction[elements]:

                var = self._extract_variable_from_input_list(reagent["symbol"])
                update_vec[var.pos] += reagent["quantity"] * multiplier

        return update_vec

    def __str__(self):
        return self.__class__.__name__ + "(" + repr(self._orig_reaction) + ")"

    def __repr__(self):
        return str(self)

    def _depends_on(self):
        self.depends_on_vector = np.zeros(len(self._pp_reaction["reagents"]),
                                          dtype=int)
        for index, reagent in enumerate(self._pp_reaction["reagents"]):
            var = self._extract_variable_from_input_list(reagent["symbol"])
            self.depends_on_vector[index] = var.pos

    def _affects(self):
        affects_vector_temp = np.nonzero(self.update_vector)
        self.affects_vector = affects_vector_temp[0]


class CommonProxyMethods:

    def __getattr__(self, attr):
        return getattr(self._obj, attr)

    def __getitem__(self, item):
        return self._obj[item]

    def __len__(self):
        return len(self._obj)

    def __str__(self):
        return self.__class__.__name__ + "(" + str(self._obj) + ")"

    def __repr__(self):
        return repr(str(self))


class VariableCollection(CommonProxyMethods):
    """Handles instances of Variable objects.

    Input: list of `string` variables.
    """

    def __init__(self, list_variables):
        self._obj = {var: Variable(var, idx)
                     for idx, var in enumerate(list_variables)}


class ParameterCollection(CommonProxyMethods):
    """Handles instances of Parameter objects.

    Input: dict of `parameter (string)`: `value` items.
    """

    def __init__(self, dict_parameters):
        self._obj = {str_param: Parameter(str_param, value)
                     for str_param, value in dict_parameters.items()}


class RateFunctionCollection(CommonProxyMethods):
    """Converts and handles RateFunction objects.

    Provides a callable method to compute the

    Input: list of functions (passed as `strings`).
    """

    def __init__(self, list_str_rate_functions, variables_collection, parameters_collection):
        self._obj = [RateFunction(str_rate_function, variables_collection, parameters_collection)
                     for str_rate_function in list_str_rate_functions]

    def __call__(self, vector):
        """Compute each function of the collection on the input numpy vector."""
        if vector.ndim != 1 or vector.shape[0] != len(self._obj):
            raise InputError("Array shapes mismatch: input vector {}, rate "
                             "functions {}.".format(vector.shape[0], len(self._obj)))

        # CHEL: the elements in the output should always be positive.
        # CHECK: should the sum of the output be equal/smaller than the system
        # size?
        return np.array(tuple(rate_func.function(vector) for rate_func in self))


class ReactionCollection(CommonProxyMethods):
    """Given a list of reactions as strings and a list of Variable(s), parse and store them."""

    def __init__(self, list_str_reactions, variables_collection):
        self._obj = [Reaction(reaction_to_be_parsed, variables_collection)
                     for reaction_to_be_parsed in list_str_reactions]

        self.update_matrix = np.stack((reac.update_vector for reac in self._obj))
        self.depends_on = np.array([reac.depends_on_vector for reac in self._obj])
        self.affects = np.array([reac.affects_vector for reac in self._obj])


class MainController:
    """Entry point class that handles the input interpretation and the simulation execution."""

    def __init__(self, dict_converted_yaml):
        self._original_yaml = dict_converted_yaml

        if len(self._original_yaml["Parameters"]) != len(self._original_yaml["Rate functions"]):
            raise InputError("The number of Parameters ({}) is different from the number of Rate"
                             " functions ({})".format(len(self._original_yaml["Parameters"]),
                                                      len(self._original_yaml["Rate functions"])))
        elif len(self._original_yaml["Initial conditions"]) != len(self._original_yaml["Species"]):
            raise InputError("There must be an initial condition for each species.")
        elif len(self._original_yaml["System size"]) != 1:
            raise InputError("The size of the system must be a single parameter. "
                             "Found: {}.".format(len(self._original_yaml["System size"])))
        elif not isinstance(self._original_yaml["Maximum simulation time"], numbers.Number):
            raise InputError("The maximum simulation time parameter (t_max) must be a number. "
                             "Found: {}.".format(self._original_yaml["Maximum simulation time"]))

        self._alg_chosen = self._original_yaml["Simulation"]
        if not isinstance(self._alg_chosen, str):
            raise InputError("The algorithm to use in the simulation must be a single string.")
        elif self._alg_chosen.lower() not in ALGORITHMS_AVAIL:
            raise InputError("The algorithm to use in the simulation must be a string in "
                             "{}.".format(", ".join((repr(alg) for alg in ALGORITHMS_AVAIL))))

        self._associate_alg(self._alg_chosen)

        self._variables = VariableCollection(self._original_yaml["Species"])
        # Treat the system size as a parameter, so it's substituted, e.g. in
        # RateFunction objects.
        self._parameters = ParameterCollection(dict(self._original_yaml["Parameters"],
                                                    **self._original_yaml["System size"]))

        self._rate_functions = RateFunctionCollection(self._original_yaml["Rate functions"],
                                                      self._variables, self._parameters)
        self._reactions = ReactionCollection(self._original_yaml["Reactions"], self._variables)
        self.update_matrix = self._reactions.update_matrix

        self._t_max = self._original_yaml["Maximum simulation time"]
        self._system_size = Parameter(*tuple(self._original_yaml["System size"].items())[0])

        # Extract the vector of initial conditions, with some checks on the
        # species provided.
        self._initial_conditions = np.empty(len(self._original_yaml["Initial conditions"]))
        for species, initial_amount in self._original_yaml["Initial conditions"].items():
            if not self._variables.get(species, False):
                raise InputError("Initial condition '{}: {}' does not match any species "
                                 "provided.".format(species, initial_amount))
            self._initial_conditions[self._variables[species].pos] = initial_amount

        self.simulation_out_times = None
        self.simulation_out_population = None

    def _associate_alg(self, str_alg):
        """Given the algorithm, set the function that matches it and a dict of optional parameters.

        Does not check again whether the algorithm in the string passed is part of the available
        algorithms, since it should have already been checked in the `__init__`.

        Save into `self._secondary_args` optional arguments that are then passed to the
        simulator.
        """
        self._secondary_args = {}

        if str_alg.lower() in ("ssa", "gillespie"):
            self._selected_alg = ssa.SSA
        elif str_alg.lower() in ("nrm", "next reaction method", "gibson bruck", "gibson-bruck"):
            self._secondary_args.update({'depends_on': self._reactions.depends_on,
                                         'affects': self._reactions.affects})
            self._selected_alg = next_reaction_method.next_reaction_method

        else:
            raise NotImplementedError("The chosen algorithm '{}' has not been "
                                      "implemented yet.".format(str_alg))

    @property
    def species(self):
        return self._variables

    def simulate(self):
        population, times = self._selected_alg(self.update_matrix,
                                               self._initial_conditions,
                                               self._rate_functions,
                                               self._t_max,
                                               **self._secondary_args)

        self.simulation_out_population, self.simulation_out_times = population, times
        return population, times

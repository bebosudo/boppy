from collections import defaultdict
import itertools
import logging
import numpy as np
import sympy as sym

from .utils import parser, misc

InputError = misc.BoppyInputError

_LOGGER = logging.getLogger(__name__)


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
    def str_var(self):
        return self._str_var

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

        self._dict_reaction = self.update_vector = None
        self.affects_vector = self.depends_on_vector = None

        self._parse_reaction()
        self._produce_update_vector()
        self._depends_on()
        self._affects()

    def _parse_reaction(self):
        self._dict_reaction = parser.parse_reaction(self._orig_reaction)
        _LOGGER.debug("Parsed string %s to Reaction object: %s",
                      self._orig_reaction,
                      self._dict_reaction)

    def _extract_variable_from_input_list(self, symbol):
        try:
            return self._variables[symbol]
        except KeyError:
            raise InputError("Unable to find reagent '{}' inside the list of variables "
                             "provided {}".format(symbol, self._variables))

    def _produce_update_vector(self):
        """Extract vector from the dictionary from reaction and the list of Variable objects.

        It searches for a match for each reagent, raising an exception if one of the variables in
        the pyparsing object is not present inside the list of Variable objects.
        """
        self.update_vector = np.zeros(len(self._variables), dtype=float)

        for symbol, quantity in itertools.chain.from_iterable(self._dict_reaction.values()):
            var = self._extract_variable_from_input_list(symbol)
            self.update_vector[var.pos] += quantity

    def __str__(self):
        return self.__class__.__name__ + "(" + repr(self._orig_reaction) + ")"

    def __repr__(self):
        return str(self)

    def _affects(self):
        """Create the vector of variables that change quantity when a reaction is executed."""
        self.affects_vector = np.nonzero(self.update_vector)[0]

    def _depends_on(self):
        """Create the vector of reactants of a reaction.

        The position in the depends_on vector is not really important, since then only the
        intersection with the affects is useful.
        """
        self.depends_on_vector = np.zeros(len(self._dict_reaction["reagents"]), dtype=int)
        for index, reagent in enumerate(self._dict_reaction["reagents"]):
            var = self._extract_variable_from_input_list(reagent.symbol)
            self.depends_on_vector[index] = var.pos


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

    Provides the callable magic method to compute each one of the converted functions on a vector.

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

        # CHECK: the elements in the output should always be positive.
        # CHECK: should the sum of the output be equal/smaller than the system size?
        return np.array(tuple(rate_func(vector) for rate_func in self))


class ReactionCollection(CommonProxyMethods):
    """Given a list of reactions as strings and a list of Variable(s), parse and store them."""

    def __init__(self, list_str_reactions, variables_collection):
        self._obj = [Reaction(reaction_to_be_parsed, variables_collection)
                     for reaction_to_be_parsed in list_str_reactions]

        self.update_matrix = np.stack((reac.update_vector for reac in self._obj))
        self.depends_on = np.array([reac.depends_on_vector for reac in self._obj])
        self.affects = np.array([reac.affects_vector for reac in self._obj])


class DependencyGraph:
    """Create the dependency graph from the vector of variables and the vector of reactans.

    The variables in the vector change quantity when a reaction is executed.
    """

    def __init__(self, affects, depends_on):
        self.graph = defaultdict(set)
        for affects_index, affects_reaction in enumerate(affects):
            for depends_on_index, depends_on_reaction in enumerate(depends_on):
                if np.intersect1d(affects_reaction, depends_on_reaction).shape[0] != 0:
                    self.graph[affects_index].add(depends_on_index)

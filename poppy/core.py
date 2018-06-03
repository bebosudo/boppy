import logging
from .utils import parser, misc
import numpy as np
import sympy as sym

LOGGER = logging.getLogger(__name__)


class Variable:
    """A Variable (or species) contains a state of the population."""

    def __init__(self, str_symbol, var_index):
        LOGGER.debug("Creating a new Variable object: %s", str_symbol)
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
        LOGGER.debug("Creating a new Parameter object: %s", str_symbol)
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

        self._pp_rate_function = parser.parse_function(self._orig_rate_function)
        LOGGER.debug("Parsed '%s' into: '%s'", self._orig_rate_function, self._pp_rate_function)

        tokenized_objs = (misc.Token(elem) for elem in self._pp_rate_function)
        rpn_tokens = parser.shunting_yard(tokenized_objs)
        LOGGER.debug("Converted '%s' to RPN sequence: '%s'", self._orig_rate_function, rpn_tokens)

        function_with_params = parser.rpn_calculator(rpn_tokens)
        LOGGER.debug("Converted RPN sequence '%s' to symbolic function: '%s'",
                     rpn_tokens, function_with_params)

        # Convert back each Parameter object to {sympy object: actual value} and substitute it.
        param_symbol_to_val = {val.symbol: val.value for val in parameters_collection.values()}
        self.sym_function = function_with_params.subs(param_symbol_to_val)
        LOGGER.debug("Substituted Parameters with their value; function '%s':", self.sym_function)

        self.lambdified = sym.lambdify(tuple(var.symbol for var in variables_collection.values()),
                                       self.sym_function)

    def function(self, arg):
        return self.lambdified(*arg)

    def __call__(self, vector):
        return self.function(vector)


class Reaction:
    """A Reaction is a combination of Variable(s) that produces other Variable(s) as output."""

    def __init__(self, str_reaction, variables_collection):
        LOGGER.debug("Creating a new Reaction object: %s", str_reaction)
        self._orig_reaction = str_reaction
        self._variables = variables_collection
        self._pp_reaction = None
        self._parse_reaction()
        self.update_vector = self._produce_update_vector()

    def _parse_reaction(self):
        self._pp_reaction = parser.parse_reaction(self._orig_reaction)
        LOGGER.debug("Parsed string %s to Reaction object: %s",
                     self._orig_reaction,
                     self._pp_reaction)

    def _extract_variable_from_input_list(self, symbol):
        try:
            return self._variables[symbol]
        except KeyError:
            raise ValueError("Unable to find reagent '{}' inside the list of variables "
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
        return self._orig_reaction


class CommonProxyMethods:

    def __getattr__(self, attr):
        return getattr(self._obj, attr)

    def __getitem__(self, item):
        return self._obj[item]

    def __len__(self):
        return len(self._obj)

    def __str__(self):
        return str(self._obj)

    def __repr__(self):
        return repr(str(self))


class VariableCollection(CommonProxyMethods):
    """Handles instances of Variable objects.

    Input: list of `string` variables.
    """

    def __init__(self, list_variables):
        self._obj = {var: Variable(var, idx) for idx, var in enumerate(list_variables)}


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
            raise ValueError("Array shapes mismatch: input vector {}, rate "
                             "functions {}.".format(vector.shape[0], len(self._obj)))

        # CHEL: the elements in the output should always be positive.
        # CHECK: should the sum of the output be equal/smaller than the system size?
        return np.array(tuple(rate_func.function(vector) for rate_func in self))


class ReactionCollection(CommonProxyMethods):
    """Given a list of reactions as strings and a list of Variable(s), parse and store them."""

    def __init__(self, str_reactions, variables_collection):
        self._obj = [Reaction(reaction_to_be_parsed, variables_collection)
                     for reaction_to_be_parsed in str_reactions]


# HyperParameters for the algorithm, kwargs to __init__

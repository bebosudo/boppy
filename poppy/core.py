import logging
from .utils import parser
import numpy as np

LOGGER = logging.getLogger(__name__)


class Variable:
    """A Variable (or species) contains a variable of the population."""

    def __init__(self, var):
        LOGGER.debug("Creating a new Variable object: %s", var)
        self._var = var

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        raise NotImplementedError("Unable to compare objects of type {} and "
                                  "{}".format(self.__class__.__name__,
                                              other.__class__.__name__))

    def __str__(self):
        return self._var

    def __repr__(self):
        return repr(str(self))


class Parameter:
    """A Parameter is the constant rate used when evolving the population."""

    def __init__(self, param):
        LOGGER.debug("Creating a new Parameter object: %s", param)
        self._param = param

    def __str__(self):
        return self._param


class Reaction:
    """A Reaction is a combination of Variable(s) that produce other certain Variable(s)."""

    def __init__(self, str_reaction, list_variables):
        LOGGER.debug("Creating a new Reaction object: %s", str_reaction)
        self._orig_reaction = str_reaction
        self._variables = list_variables
        self._parse_reaction()
        self.update_vector = self._produce_update_vector()

    def _parse_reaction(self):
        self._pp_reaction = parser.evaluate_reaction(self._orig_reaction)
        LOGGER.debug("Parsed string %s to Reaction object: %s",
                     self._orig_reaction,
                     self._pp_reaction)

    def _extract_variable_from_input_list(self, symbol):
        var = Variable(symbol)
        try:
            return self._variables.index(var)
        except ValueError:
            raise ValueError("Unable to find reagent '{}' inside the list of variables "
                             "provided {}".format(var, self._variables))

    def _produce_update_vector(self):
        """Create the update vector from the parsed reaction and the list of Variable objects.

        It searches for a match for each reagent, raising an exception if one of the variables in
        the pyparsing object is not present inside the list of Variable objects.
        """

        update_vec = np.zeros_like(self._variables, dtype=float)

        # Each reagent used in the reaction decreases the quantity available, while the products
        # increase it.
        for elements, multiplier in (("reagents", -1), ("products", 1)):
            for reagent in self._pp_reaction[elements]:
                pos = self._extract_variable_from_input_list(reagent["symbol"])

                update_vec[pos] = reagent["quantity"] * multiplier

        return update_vec

    def __str__(self):
        return self._orig_reaction

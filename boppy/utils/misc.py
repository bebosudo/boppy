from collections import namedtuple
import numbers
import operator
import sympy as sym


class BoppyInputError(Exception):
    pass


_func_tuple = namedtuple("FunctionAndNumArgs", ("python_function", "number_args"))

# Python operators are automatically converted to sympy operators.
_ADD_OPS = {"+": sym.Add, "-": operator.sub}
_MUL_OPS = {"*": sym.Mul, "/": operator.truediv}

ARITH_OPS = _ADD_OPS.copy()
ARITH_OPS.update(_MUL_OPS)

AVAIL_FUNCTIONS = {"abs":   _func_tuple(sym.Abs, 1),
                   "acos":  _func_tuple(sym.acos, 1),
                   "acosh": _func_tuple(sym.acosh, 1),
                   "asin":  _func_tuple(sym.asin, 1),
                   "asinh": _func_tuple(sym.asinh, 1),
                   "atan":  _func_tuple(sym.atan, 1),
                   "atan2": _func_tuple(sym.atan2, 2),
                   "atanh": _func_tuple(sym.atanh, 1),
                   "ceil":  _func_tuple(sym.ceiling, 1),
                   "cos":   _func_tuple(sym.cos, 1),
                   "cosh":  _func_tuple(sym.cosh, 1),
                   "exp":   _func_tuple(sym.exp, 1),
                   "floor": _func_tuple(sym.floor, 1),
                   "fmod":  _func_tuple(sym.Mod, 2),
                   "log":   _func_tuple(sym.log, 1),
                   "log10": _func_tuple(lambda x: sym.log(x, 10), 1),
                   "max":   _func_tuple(sym.Max, -1),
                   "min":   _func_tuple(sym.Min, -1),
                   "pow":   _func_tuple(sym.Pow, 2),
                   "round": _func_tuple(sym.ceiling, 1),
                   "sin":   _func_tuple(sym.sin, 1),
                   "sinh":  _func_tuple(sym.sinh, 1),
                   "sqrt":  _func_tuple(sym.sqrt, 1),
                   "sum":   _func_tuple(sym.Sum, -1),
                   "tan":   _func_tuple(sym.tan, 1),
                   "tanh":  _func_tuple(sym.tanh, 1)
                   }


class Token:
    """Assigns each string a meaning, such as `identifier', `separator', `operator', etc.

    Gives each of them a precedence (an order of relevance among operators) needed in the
    shunting yard algorithm to compute the result.

    So far, we provide the pow function in substitution of the ^ power symbol, so we don't need to
    handle right-associativity.
    """

    def __init__(self, elem):
        self._token_element = elem

        if self._token_element in _ADD_OPS:
            self.precedence = 1
        elif self._token_element in _MUL_OPS:
            self.precedence = 2
        else:
            self.precedence = 0

        self.is_number = isinstance(elem, numbers.Number)
        self.is_function = elem in AVAIL_FUNCTIONS
        self.is_operator = elem in ARITH_OPS
        self.is_leftpar = elem == "("
        self.is_rightpar = elem == ")"
        self.is_comma = elem == ","
        self.is_variable = not any((self.is_number, self.is_function, self.is_operator,
                                    self.is_leftpar, self.is_rightpar, self.is_comma))

    @property
    def value(self):
        return self._token_element

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.precedence == other.precedence
        return False

    def __lt__(self, other):
        if isinstance(self, other.__class__):
            return self.precedence < other.precedence
        return NotImplemented

    def __gt__(self, other):
        if isinstance(self, other.__class__):
            return self.precedence > other.precedence
        return NotImplemented

    def __ne__(self, other):
        return not self == other

    def __ge__(self, other):
        return not self < other

    def __le__(self, other):
        return not self > other

    def __str__(self):
        return str(self._token_element)

    def __repr__(self):
        return repr(str(self))


# This is a symbol used in the shunting yard and RPN calculator to separate arguments of functions
# that accept multiple args from arguments of functions with defined number of args.
FUNC_ARGS_SEPARATOR = Token("|")

import pyparsing as pp
import operator
import math
from collections import deque


DIGITS_PRECISION = 14

ADD_OPS = {"+", "-"}
MUL_OPS = {"*", "/"}

# If you edit this char below, don't use an arithmetic symbol or a function name
# like the ones in ADD_OPS, MUL_OPS or AVAIL_FUNCTIONS dictionaries!
FUNC_ARGS_SEPARATOR = "."


AVAIL_FUNCTIONS = {"abs": (abs, 1),
                   "acos": (math.acos, 1),
                   "acosh": (math.acosh, 1),
                   "asin": (math.asin, 1),
                   "asinh": (math.asinh, 1),
                   "atan": (math.atan, 1),
                   "atan2": (math.atan2, 2),
                   "atanh": (math.atanh, 1),
                   "ceil": (math.ceil, 1),
                   "cos": (math.cos, 1),
                   "cosh": (math.cosh, 1),
                   "decbin": (bin, 1),
                   "decoct": (oct, 1),
                   "deg2rad": (math.radians, 1),
                   "exp": (math.exp, 1),
                   "expm1": (lambda x: math.exp(x) - 1, 1),
                   "floor": (math.floor, 1),
                   "fmod": (math.fmod, 2),
                   "is_finite": (lambda x: not math.isinf, 1),
                   "is_infinite": (math.isinf, 1),
                   "is_nan": (math.isnan, 1),
                   "log10": (math.log10, 1),
                   "log": (math.log, 1),
                   "log1p": (math.log1p, 1),
                   "max": (max, -1),
                   "min": (min, -1),
                   "octdec": (lambda x: int(x, 8), 1),
                   "pow": (math.pow, 2),
                   "rad2deg": (math.degrees, 1),
                   # TODO: check if the rand func in moodle works exactly like this
                   # "rand": (lambda x: random.randint(-sys.maxint - 1,
                   #                                   sys.maxint), 1),
                   "round": (round, 1),
                   "sin": (math.sin, 1),
                   "sinh": (math.sinh, 1),
                   "sqrt": (math.sqrt, 1),
                   "tan": (math.tan, 1),
                   "tanh": (math.tanh, 1)
                   }


class Token:
    """Assign each string a meaning, such as `identifier', `separator', `operator', etc.

    Gives each of them a precedence (an order of relevance among operators) needed in the
    shunting yard algorithm to compute the result.

    So far, we provide the pow function in substitution of the ^ power symbol, so we don't need to
    handle right-associativity.
    """

    def __init__(self, elem):
        self._token_element = elem

        if self._token_element in ADD_OPS:
            self.precedence = 1
        elif self._token_element in MUL_OPS:
            self.precedence = 2
        else:
            self.precedence = 0

    @property
    def internal_value(self):
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


class CommonParserComponents:
    """Collects basic syntax components that can be reused to build different parsers.

    The EBNF grammar is stored at "docs/grammar.ebnf".

    https://infohost.nmt.edu/tcc/help/pubs/pyparsing/web/struct-results-name.html

    TODO: fix `variable', so that something like `-x' can be parsed.
    """

    def __init__(self):
        self.digits = pp.Word(pp.nums)
        self.plus_or_minus = pp.oneOf("+ -")
        self.opt_plus_minus = pp.Optional(self.plus_or_minus)
        self.mul_or_div = pp.oneOf("* /")
        self.point = pp.Word(".")
        self.left_par = pp.Literal("(")
        self.right_par = pp.Literal(")")

        self.unsigned_int = self.digits
        self.signed_int = pp.Combine(self.plus_or_minus + self.unsigned_int)

        self.opt_signed_int = (pp.Combine(self.opt_plus_minus + self.unsigned_int)
                               .setParseAction(lambda el: int(el[0])))

        self.float_num = pp.Combine(self.opt_plus_minus +
                                    ((self.unsigned_int + self.point + pp.Optional(self.unsigned_int)) ^
                                     (self.point + self.unsigned_int)
                                     ) +
                                    pp.Optional(pp.CaselessLiteral("e") + self.opt_signed_int)
                                    ).setParseAction(lambda el: float(el[0]))

        self.real_num = (self.float_num ^ self.opt_signed_int)

        self.variable_name = pp.Word(pp.alphas + "_", pp.alphas + pp.nums + "_")
        # self.variable = pp.Combine(self.opt_plus_minus +
        #                            self.variable_name
        #                            ).setResultsName("variable")


class FunctionParser(CommonParserComponents):
    """A poppy valid Function has (almost) the same syntax of a python function."""

    def __init__(self):
        super(FunctionParser, self).__init__()

        self.add_op = pp.Forward()
        self.mul_op = pp.Forward()
        self.expr = pp.Forward()

        self.function = (self.variable_name +
                         self.left_par +
                         self.add_op +
                         pp.ZeroOrMore("," + self.add_op) +
                         self.right_par
                         )

        self.add_op << (self.mul_op + pp.ZeroOrMore(self.plus_or_minus + self.mul_op))
        self.mul_op << (self.expr + pp.ZeroOrMore(self.mul_or_div + self.expr)
                        )
        self.expr << ((self.opt_plus_minus + self.left_par + self.add_op + self.right_par) ^
                      self.real_num ^
                      self.function ^
                      self.variable_name
                      )

    def __getattr__(self, attr):
        # self.add_op.setDebug()
        return getattr(self.add_op(), attr)


class ReactionParser(CommonParserComponents):

    def __init__(self):
        super(ReactionParser, self).__init__()

        reaction_symbol = pp.Suppress("=>")
        reagents_sum_sym = pp.Suppress("+")

        # A missing quantity must be interpreted as 1 unit.
        qtt_with_sym = pp.Group(pp.Optional(self.real_num, default=1).setResultsName("quantity") +
                                self.variable_name.setResultsName("symbol"))

        self.reaction = (pp.Group(qtt_with_sym +
                                  pp.ZeroOrMore(reagents_sum_sym + qtt_with_sym)
                                  ).setResultsName("reagents") +
                         reaction_symbol +
                         pp.Group(qtt_with_sym +
                                  pp.ZeroOrMore(reagents_sum_sym + qtt_with_sym)
                                  ).setResultsName("products")
                         )

    def __getattr__(self, attr):
        return getattr(self.reaction(), attr)


_FUNCTION_GRAMMAR = FunctionParser()
_REACTION_GRAMMAR = ReactionParser()


def parse_reaction(str_reaction):
    return _REACTION_GRAMMAR.parseString(str_reaction)


def parse_function(str_function):
    return _FUNCTION_GRAMMAR.parseString(str_function)


def shunting_yard(list_of_tokens):
    """Given a list of numbers and Token(s), return a postfix notation ordered stack.

    (e.g. "3 + 4 * 2 / ( 1 - 5 )" becomes: "3 4 2 * 1 5 - / +")

    https://en.wikipedia.org/wiki/Shunting-yard_algorithm
    http://www.reedbeta.com/blog/2011/12/11/the-shunting-yard-algorithm/
    """

    out_queue = deque()
    op_stack = deque()

    for curr_token in list_of_tokens:
        if curr_token.is_number:
            out_queue.append(curr_token)
        elif curr_token.is_function:
            op_stack.append(curr_token)
        elif curr_token.is_operator:
            while (op_stack and
                   (op_stack[-1].is_function or op_stack[-1] >= curr_token) and
                    not op_stack[-1].is_leftpar):
                out_queue.append(op_stack.pop())
            op_stack.append(curr_token)
        elif curr_token.is_leftpar:
            op_stack.append(curr_token)
        elif curr_token.is_rightpar:
            while not op_stack[-1].is_leftpar:
                out_queue.append(op_stack.pop())
            op_stack.pop()
    # There should not be any parentheses mismatch, since we perform an initial parsing, and
    # therefore we don't allow any ill-formed elements at a previous step.
    op_stack.reverse()
    out_queue.extend(op_stack)

    return out_queue


def rpn_calculator(rpn_tokens):
    """Convert the RPN expression inside `rpn_tokens' into a sympy symbolic expression.

    https: // en.wikipedia.org/wiki/Reverse_Polish_notation
    """
    pass

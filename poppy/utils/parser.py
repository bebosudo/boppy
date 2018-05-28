import pyparsing as pp
import operator
import math

DIGITS_PRECISION = 14

# moodle_funcs_web = ("https://docs.moodle.org/22/en/Calculated_question_"
#                     "type#Available_functions")


ADD_OPS = {"+": (operator.add, 2),
           "-": (operator.sub, 2)
           }

MUL_OPS = {"*": (operator.mul, 2),
           "/": (operator.truediv, 2)
           }

# If you edit this char below, don't use an arithmetic symbol or a function name
# like the ones in ADD_OPS, MUL_OPS or AVAIL_FUNCTIONS dictionaries!
FUNC_ARGS_SEPARATOR = "."

# Just an extended dict containing the above two arithmetic dicts.
ARITH_OPERATIONS = {}
ARITH_OPERATIONS.update(ADD_OPS)
ARITH_OPERATIONS.update(MUL_OPS)


# The "pi()" function is managed internally at the lexer (see the `syntax' function),
# because it's the only function without args.
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
    """This class manages arithmetic operators like + - * / and parenthesis.

    It gives them a precedence (an order of relevance between operators) needed
    later in the shunting yard algorithm to compute the result.
    We use the pow function in substitution of the ^ power symbol, so we don't need to handle
    right-associativity.
    """

    def __init__(self, symbol_char):
        self.symbol_char = symbol_char[0]

        # The order (or "precedence") means simply the precedence an operator has among others.
        if self.symbol_char in ADD_OPS:
            self.precedence = 1
        elif self.symbol_char in MUL_OPS:
            self.precedence = 2
        else:
            self.precedence = 0

    def __hash__(self):
        return hash(self.symbol_char)

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.symbol_char == other.symbol_char
        return False

    def __lt__(self, other):
        if isinstance(self, other.__class__):
            return self.precedence < other.precedence
        raise NotImplementedError()

    def __gt__(self, other):
        if isinstance(self, other.__class__):
            return self.precedence > other.precedence
        raise NotImplementedError()

    def __ne__(self, other):
        return not self == other

    def __ge__(self, other):
        return not self < other

    def __le__(self, other):
        return not self > other

    def __str__(self):
        return self.symbol_char

    def __repr__(self):
        return repr(str(self))


class BasicSyntax:
    """This class collects basic syntax components.

    The EBNF grammar is stored at "docs/grammar.ebnf".

    https://infohost.nmt.edu/tcc/help/pubs/pyparsing/web/struct-results-name.html
    """

    def __init__(self):
        self.digits = pp.Word(pp.nums).setName("digits")
        self.plus_or_minus = pp.oneOf("+ -").setParseAction(self._tokenize_it).setName("+or-")
        self.opt_plus_minus = pp.Optional(self.plus_or_minus)
        self.mul_or_div = pp.oneOf("* /").setParseAction(self._tokenize_it).setName("*or/")
        self.point = pp.Word(".").setName("point")
        self.left_par = pp.Literal("(").setParseAction(self._tokenize_it).setName("left par")
        self.right_par = pp.Literal(")").setParseAction(self._tokenize_it).setName("right par")

        self.unsigned_int = self.digits
        self.signed_int = pp.Combine(self.plus_or_minus + self.unsigned_int).setName("signed int")

        self.opt_signed_int = (pp.Combine(self.opt_plus_minus + self.unsigned_int)
                               .setParseAction(lambda el: int(el[0]))
                               .setName("optionally signed int"))
        self.float_num = pp.Combine(self.opt_plus_minus +
                                    (self.unsigned_int + self.point + pp.Optional(self.unsigned_int)) ^ (self.point + self.unsigned_int) +
                                    pp.Optional(pp.CaselessLiteral("e") + self.opt_signed_int)
                                    ).setParseAction(lambda el: float(el[0])).setName("float number")
        self.pi = pp.Combine(self.opt_plus_minus +
                             pp.Literal("pi()").setParseAction(lambda el: math.pi)
                             ).setParseAction(lambda el: float(el[0])).setName("greek pi")

        self.real_num = (self.float_num ^ self.opt_signed_int ^ self.pi).setName("real number")

        self.variable_name = pp.Word(pp.alphas + "_", pp.alphas + pp.nums + "_")
        self.variable = pp.Combine(self.opt_plus_minus + self.variable_name).setName("variable")

    def _tokenize_it(self, _op):
        return Token(_op)


class RateFunctionSyntax(BasicSyntax):

    def __init__(self):
        super(RateFunctionSyntax, self).__init__()
        self.add_op = pp.Forward()
        self.mul_op = pp.Forward()
        self.expr = pp.Forward()

        self.function = (self.variable + self.left_par + self.add_op +
                         pp.ZeroOrMore("," + self.add_op) + self.right_par).setName("function")

        self.add_op << (self.mul_op + pp.ZeroOrMore(self.plus_or_minus + self.mul_op)
                        ).setName("add operator")
        self.mul_op << (self.expr + pp.ZeroOrMore(self.mul_or_div + self.expr)
                        ).setName("mul operator")
        self.expr << ((self.opt_plus_minus + self.left_par + self.add_op + self.right_par) ^
                      self.function ^
                      self.real_num
                      ).setName("atomic expression")

    def __getattr__(self, attr):
        # self.add_op.setDebug()
        return getattr(self.add_op(), attr)


class ReactionSyntax(BasicSyntax):

    def __init__(self):
        super(ReactionSyntax, self).__init__()

        reaction_symbol = pp.Literal("=>").suppress()
        reagents_sum_sym = pp.Literal("+").suppress()

        # A missing quantity must be interpreted as 1 unit.
        qtt_with_sym = pp.Group(pp.Optional(self.real_num, default=1).setResultsName("quantity") +
                                self.variable.setResultsName("symbol"))

        self.reaction = (pp.Group(qtt_with_sym +
                                  pp.OneOrMore(reagents_sum_sym + qtt_with_sym)
                                  ).setResultsName("reagents") +
                         reaction_symbol +
                         pp.Group(qtt_with_sym +
                                  pp.ZeroOrMore(reagents_sum_sym + qtt_with_sym)
                                  ).setResultsName("products")
                         )

    def __getattr__(self, attr):
        return getattr(self.reaction(), attr)


_RATE_FUNCTION_GRAMMAR = RateFunctionSyntax()
_REACTION_GRAMMAR = ReactionSyntax()


def evaluate_reaction(str_reaction):
    return _REACTION_GRAMMAR.parseString(str_reaction)


if __name__ == "__main__":
    reac = "X + 2y => X + 4 z "
    eval_reac = evaluate_reaction(reac)

    # list_vars
    # upd = produce_update_vector(eval_reac, )
    print(repr(reac), "becomes", evaluate_reaction(reac))

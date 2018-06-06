import pyparsing as pp
from collections import deque
import sympy as sym

from .misc import ARITH_OPS, AVAIL_FUNCTIONS, FUNC_ARGS_SEPARATOR

DIGITS_PRECISION = 14


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
    """A boppy valid Function has (almost) the same syntax of a python function."""

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
        if curr_token.is_number or curr_token.is_variable:
            out_queue.append(curr_token)

        elif curr_token.is_function:
            # If a function, such as `max()', accepts multiple arguments, insert a separator
            # between the internal function arguments and the external elements.
            if AVAIL_FUNCTIONS[curr_token.value].number_args < 0:
                out_queue.append(FUNC_ARGS_SEPARATOR)

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

        # Every other symbol (such as `,') is discarded.

    # There should not be any parentheses mismatch, since we perform an initial parsing, and
    # therefore we don't allow any ill-formed elements at a previous step.
    op_stack.reverse()
    out_queue.extend(op_stack)

    return out_queue


def rpn_calculator(rpn_tokens):
    """Convert the RPN expression inside `rpn_tokens' into a sympy symbolic expression.

    https: // en.wikipedia.org/wiki/Reverse_Polish_notation
    """
    op_stack = deque()

    _separator_to_drop = sym.Symbol(FUNC_ARGS_SEPARATOR.value)

    for curr_token in rpn_tokens:

        if curr_token.is_operator or curr_token.is_function:

            if curr_token.is_function and AVAIL_FUNCTIONS[curr_token.value].number_args < 0:
                _op = AVAIL_FUNCTIONS[curr_token.value].python_function
                # `iter(func, sentinel)' drops the sentinel, so we need to use a different strategy
                # when dealing with a fixed number of chars.
                _args = iter(op_stack.pop, _separator_to_drop)

            else:
                if curr_token.value in ARITH_OPS:
                    _op = ARITH_OPS[curr_token.value]
                    num_args = 2
                elif curr_token.value in AVAIL_FUNCTIONS:
                    _op = AVAIL_FUNCTIONS[curr_token.value].python_function
                    num_args = AVAIL_FUNCTIONS[curr_token.value].number_args
                else:
                    raise ValueError("Operator '{}' marked as function, but no functions are "
                                     "associated to that name.".format(curr_token.value))

                _args = (op_stack.pop() for _ in range(num_args))

            op_stack.append(_op(*reversed(tuple(_args))))

        elif curr_token.is_variable:
            op_stack.append(sym.Symbol(curr_token.value))

        elif curr_token.is_number:
            op_stack.append(sym.Float(curr_token.value))

    return op_stack.pop()

from copy import deepcopy
import re
import rtamt


# ── rtamt: syntax validation only ─────────────────────────────────────────────

def _validate_syntax(formula_str: str) -> None:
    """Raise if formula_str is not valid rtamt STL syntax."""
    spec = rtamt.StlDiscreteTimeSpecification()
    spec.declare_var('x', 'float')
    spec.declare_var('y', 'float')
    spec.spec = formula_str
    spec.parse()


# ── Boolean evaluator ─────────────────────────────────────────────────────────
# rtamt uses robustness semantics, where equality predicates give robustness = 0
# at the exact boundary — making it impossible to distinguish satisfied from
# violated for integer-valued trajectories.  We use rtamt only for parsing and
# do evaluation ourselves with pure boolean semantics.

def _tokenize(s: str) -> list:
    patterns = [
        ('AND',    r'\band\b'),
        ('OR',     r'\bor\b'),
        ('NOT',    r'\bnot\b'),
        ('G',      r'\bG\b'),
        ('F',      r'\bF\b'),
        ('U',      r'\bU\b'),
        ('OP',     r'==|!=|>=|<=|>|<'),
        ('LPAREN', r'\('),
        ('RPAREN', r'\)'),
        ('NUM',    r'-?\d+(?:\.\d+)?'),
        ('SIG',    r'\b[xy]\b'),
    ]
    tok_re = '|'.join(f'(?P<{name}>{pat})' for name, pat in patterns)
    return [(m.lastgroup, m.group()) for m in re.finditer(tok_re, s)]


class _BoolEval:
    """Recursive-descent boolean evaluator for RTAmt-style STL formulas.

    All methods return a list of n booleans (one per trajectory step).
    Temporal operators G and F broadcast a single truth value to all n slots,
    so they compose correctly when nested.
    """

    def __init__(self, tokens: list, xs: list, ys: list):
        self.tokens = tokens
        self.pos    = 0
        self.xs     = [float(v) for v in xs]
        self.ys     = [float(v) for v in ys]
        self.n      = len(xs)

    def _peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else ('EOF', '')

    def _consume(self, kind=None):
        tok = self.tokens[self.pos]
        if kind and tok[0] != kind:
            raise ValueError(f"Expected {kind!r}, got {tok!r}")
        self.pos += 1
        return tok

    def parse(self) -> list:
        result = self._or()
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token: {self.tokens[self.pos]!r}")
        return result

    def _or(self) -> list:
        left = self._and()
        while self._peek()[0] == 'OR':
            self._consume('OR')
            right = self._and()
            left = [a or b for a, b in zip(left, right)]
        return left

    def _and(self) -> list:
        left = self._until()
        while self._peek()[0] == 'AND':
            self._consume('AND')
            right = self._until()
            left = [a and b for a, b in zip(left, right)]
        return left

    def _until(self) -> list:
        left = self._unary()
        if self._peek()[0] == 'U':
            self._consume('U')
            right = self._unary()
            result = []
            for t in range(self.n):
                sat = any(
                    right[s] and all(left[t:s])
                    for s in range(t, self.n)
                )
                result.append(sat)
            return result
        return left

    def _unary(self) -> list:
        kind, val = self._peek()
        if kind == 'NOT':
            self._consume('NOT')
            self._consume('LPAREN')
            inner = self._or()
            self._consume('RPAREN')
            return [not v for v in inner]
        if kind == 'G':
            self._consume('G')
            self._consume('LPAREN')
            inner = self._or()
            self._consume('RPAREN')
            v = all(inner)
            return [v] * self.n
        if kind == 'F':
            self._consume('F')
            self._consume('LPAREN')
            inner = self._or()
            self._consume('RPAREN')
            v = any(inner)
            return [v] * self.n
        if kind == 'LPAREN':
            self._consume('LPAREN')
            inner = self._or()
            self._consume('RPAREN')
            return inner
        return self._predicate()

    def _predicate(self) -> list:
        kind, val = self._peek()
        if kind == 'SIG':
            self._consume('SIG')
            signal = val
            _, op = self._consume('OP')
            _, num_str = self._consume('NUM')
            num = float(num_str)
            sig_vals = self.xs if signal == 'x' else self.ys
            return [_cmp(v, op, num) for v in sig_vals]
        if kind == 'NUM':
            self._consume('NUM')
            num = float(val)
            _, op = self._consume('OP')
            _, sig = self._consume('SIG')
            sig_vals = self.xs if sig == 'x' else self.ys
            return [_cmp(num, op, v) for v in sig_vals]
        raise ValueError(f"Expected predicate, got {kind!r} {val!r}")


def _cmp(a, op, b) -> bool:
    if op == '==': return a == b
    if op == '!=': return a != b
    if op == '>=': return a >= b
    if op == '<=': return a <= b
    if op == '>':  return a > b
    if op == '<':  return a < b
    raise ValueError(f"Unknown operator: {op!r}")


def _eval_boolean(formula_str: str, xs: list, ys: list) -> bool:
    """Evaluate formula_str with boolean semantics over the trajectory."""
    tokens = _tokenize(formula_str)
    ev = _BoolEval(tokens, xs, ys)
    result = ev.parse()
    return result[0]  # time-0 value; G/F broadcast, so this is the full-trace result


# ── check function ─────────────────────────────────────────────────────────────

def check_stl_math(sample: dict, formula_str: str) -> str | None:
    """Return None on success, or an error string describing the first failure."""

    try:
        _validate_syntax(formula_str)
    except Exception as e:
        return f"Failed to parse STL formula: {e}"

    x_true = [coord[0] for coord in sample['solution_coordinates']]
    y_true = [coord[1] for coord in sample['solution_coordinates']]

    should_pass = [(x_true, y_true)]

    # A path that ends back at the start instead of the goal.
    x_false = deepcopy(x_true)
    y_false = deepcopy(y_true)
    x_false[-1] = x_false[0]
    y_false[-1] = y_false[0]
    should_fail = [(x_false, y_false, "Doesn't reach the goal")]

    # For each obstacle cell, a path that visits that cell.
    for i, row in enumerate(sample['world']):
        for j, cell in enumerate(row):
            if cell == 1:
                x_obstacle = x_true + [i]
                y_obstacle = y_true + [j]
                should_fail.append((x_obstacle, y_obstacle, f"Hits an obstacle at ({i}, {j})."))

    for xs, ys in should_pass:
        try:
            holds = _eval_boolean(formula_str, xs, ys)
        except Exception as e:
            return f"Failed to evaluate the STL formula: {e}"
        if not holds:
            traj_str = " -> ".join(f"({xi}, {yi})" for xi, yi in zip(xs, ys))
            return f"The formula is not satisfied for the ground-truth trajectory {traj_str}."

    for xs, ys, reason in should_fail:
        try:
            holds = _eval_boolean(formula_str, xs, ys)
        except Exception as e:
            return f"Failed to evaluate the STL formula: {e}"
        if holds:
            traj_str = " -> ".join(f"({xi}, {yi})" for xi, yi in zip(xs, ys))
            return f"The formula should be false for trajectory {traj_str}. Reason: {reason}"

    return None


if __name__ == "__main__":

    import unittest

    class TestParseStlMath(unittest.TestCase):
        # Minimal 3x3 synthetic sample.
        SAMPLE_3x3 = {
            'world': [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            'solution_coordinates': [[0, 0], [0, 1], [0, 2]],
            'nl_description': 'You are in a 3 by 3 world. There is an obstacle at (1, 1). Go from (0, 0) to (0, 2)',
        }

        # Real sample from 1_goals_test_seen_6x6_samples.json (index 0).
        # World: 6x6, obstacle at (5,3), start (3,1), goal (1,3).
        # Solution: (3,1) -> (2,1) -> (1,1) -> (1,2) -> (1,3)
        SAMPLE_6x6 = {
            'world': [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 3, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ],
            'solution_coordinates': [[3, 1], [2, 1], [1, 1], [1, 2], [1, 3]],
            'nl_description': 'You are in a 6 by 6 world. There are obstacles that you have to avoid at: (5,3). Go from (3,1) to (1,3)',
            'agent_as_a_point': 'up up right right',
        }

        # ── 3x3 tests ─────────────────────────────────────────────────────────

        def test_correct_formula_3x3(self):
            formula = "G(not(x == 1 and y == 1)) and F(x == 0 and y == 2)"
            self.assertIsNone(check_stl_math(self.SAMPLE_3x3, formula))

        def test_syntax_error_3x3(self):
            formula = "always x == 1"  # 'always' not valid in rtamt
            self.assertIsNotNone(check_stl_math(self.SAMPLE_3x3, formula))

        def test_no_goal_3x3(self):
            # No goal requirement — accepts the wrong-endpoint path.
            formula = "G(not(x == 1 and y == 1))"
            self.assertIsNotNone(check_stl_math(self.SAMPLE_3x3, formula))

        def test_no_obstacle_avoidance_3x3(self):
            # No obstacle check — accepts obstacle-visiting path.
            formula = "F(x == 0 and y == 2)"
            self.assertIsNotNone(check_stl_math(self.SAMPLE_3x3, formula))

        # ── 6x6 real-sample tests ─────────────────────────────────────────────

        def test_correct_formula_6x6(self):
            # Correct formula: avoid obstacle (5,3), reach goal (1,3).
            formula = "G(not(x == 5 and y == 3)) and F(x == 1 and y == 3)"
            self.assertIsNone(check_stl_math(self.SAMPLE_6x6, formula))

        def test_no_obstacle_avoidance_6x6(self):
            # No obstacle check — accepts path that visits (5,3).
            formula = "F(x == 1 and y == 3)"
            err = check_stl_math(self.SAMPLE_6x6, formula)
            self.assertIsNotNone(err)
            self.assertIn("(5, 3)", err)

        def test_wrong_goal_6x6(self):
            # Obstacle avoidance correct but goal wrong — should fail on ground-truth path.
            formula = "G(not(x == 5 and y == 3)) and F(x == 0 and y == 0)"
            err = check_stl_math(self.SAMPLE_6x6, formula)
            self.assertIsNotNone(err)
            self.assertIn("ground-truth", err)

    unittest.main()

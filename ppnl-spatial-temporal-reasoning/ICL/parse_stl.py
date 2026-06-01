from copy import deepcopy
from stl_def import *

def evaluate_stl(stl: str, x: Trajectory, y: Trajectory):
    """Evaluate an STL string that is either a bare expression or a safe_paths(x, y) function def."""
    # make sre we run the function in the exec
    stl += '\nsafe_paths(x,y)' # this is a hack to make sure we run the function in the exec environment

    globals_dict = {
        "x": x,
        "y": y,
        "Always": Always,
        "Eventually": Eventually,
        "Until": Until,
        "And": And,
        "Or": Or,
        "Not": Not,
        "Implies": Implies,
        "Trajectory": Trajectory,
    }
    try:
        return eval(stl, globals_dict)
    except SyntaxError:
        local_ns = {}
        exec(stl, globals_dict, local_ns)
        fn = local_ns.get("safe_paths") or globals_dict.get("safe_paths")
        if fn is None:
            raise ValueError("STL code must be an expression or define a safe_paths(x, y) function")
        return fn(x, y)


def check_stl(sample: dict, stl: str) -> str | None:
    """Return None on success, or an error string describing the first failure."""

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
                x_obstacle = [i] +  x_true
                y_obstacle = [j] + y_true
                should_fail.append((x_obstacle, y_obstacle, f"Hits an obstacle at ({i}, {j})."))

    # Typecheck against the first passing trajectory.
    traj_x = Trajectory(should_pass[0][0])
    traj_y = Trajectory(should_pass[0][1])
    try:
        stl_expr = evaluate_stl(stl, traj_x, traj_y)
    except Exception as e:
        return f"Failed to evaluate the STL expression. Error: {e}"

    try:
        stl_expr.typecheck()
    except TypeError as e:
        return f"The STL expression is not valid. Error: {e}"
    except Exception as e:
        raise e

    for xs, ys in should_pass:
        traj_x = Trajectory(xs)
        traj_y = Trajectory(ys)
        stl_expr = evaluate_stl(stl, traj_x, traj_y)
        
        if not all(stl_expr()):
            traj_str = " -> ".join([f"({xi}, {yi})" for xi, yi in zip(xs, ys)])
            return f"The STL expression is not true for the ground-truth trajectory {traj_str}."

    for xs, ys, reason in should_fail:
        traj_x = Trajectory(xs)
        traj_y = Trajectory(ys)
        stl_expr = evaluate_stl(stl, traj_x, traj_y)
        if all(stl_expr()):
            traj_str = " -> ".join([f"({xi}, {yi})" for xi, yi in zip(xs, ys)])
            return f"The STL expression should be false for trajectory {traj_str}. Reason: {reason}"

    return None


if __name__ == "__main__":

    import unittest

    class TestParseStl(unittest.TestCase):
        def test_expression_stl(self):
            # 3x3 world, obstacle at (1,1), path (0,0)->(0,1)->(0,2).
            # x encodes rows, y encodes cols.
            sample = {
                'world': [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                'solution_coordinates': [[0, 0], [0, 1], [0, 2]],
                'nl_description': 'You are in a 3 by 3 world. There is an obstacle at (1, 1). Go from (0, 0) to (0, 2)',
            }
            stl = "And(Always(Not(And(x == 1, y == 1))), Eventually(And(x == 0, y == 2)))"
            self.assertIsNone(check_stl(sample, stl))

        def test_function_stl(self):
            # Same scenario expressed as a safe_paths function def.
            sample = {
                'world': [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                'solution_coordinates': [[0, 0], [0, 1], [0, 2]],
                'nl_description': 'You are in a 3 by 3 world. There is an obstacle at (1, 1). Go from (0, 0) to (0, 2)',
            }
            stl = (
                "def safe_paths(x, y):\n"
                "    obstacle_free = Always(Not(And(x == 1, y == 1)))\n"
                "    reach_goal = Eventually(And(x == 0, y == 2))\n"
                "    return And(obstacle_free, reach_goal)\n"
            )
            self.assertIsNone(check_stl(sample, stl))

        def test_wrong_stl_no_goal(self):
            # A formula with no goal requirement accepts the wrong-endpoint path -> rejected.
            sample = {
                'world': [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                'solution_coordinates': [[0, 0], [0, 1], [0, 2]],
                'nl_description': 'You are in a 3 by 3 world. There is an obstacle at (1, 1). Go from (0, 0) to (0, 2)',
            }
            stl = "Always(Not(And(x == 1, y == 1)))"  # no goal → accepts bad paths
            self.assertIsNotNone(check_stl(sample, stl))

    unittest.main()

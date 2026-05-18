"""
Convert code_form.py JSONL output into the format expected by executor-point-sg.py.

Each move_x(d) / move_y(d) call is expanded into d repeated action tokens:
  move_x(+d) -> "right" * d
  move_x(-d) -> "left"  * d
  move_y(+d) -> "down"  * d
  move_y(-d) -> "up"    * d

Supports both output formats from code_form.py:
  - Legacy (single attempt): record has a top-level "response" field.
  - Multi-try: record has an "attempts" list; the first attempt whose
    "error" is null is used. If all failed the last attempt is used.

Each output record includes:
  "error":          null on success, or a plain-English error description.
  "successful_try": 1-based try number that succeeded, or null if all failed.
"""
import json
import re
import sys


def parse_response(response):
    """
    Parse move_x/move_y calls from the LLM response.
    Returns (action_str, skipped) where skipped is a list of calls whose
    arguments could not be parsed as integers (e.g. variables like N).
    """
    actions = []
    skipped = []
    for match in re.finditer(r'move_([xy])\(([^)]+)\)', response):
        axis, arg = match.group(1), match.group(2).strip()
        try:
            dist = int(arg)
        except ValueError:
            skipped.append(f"move_{axis}({arg})")
            continue
        if axis == 'x':
            actions.extend(['right' if dist > 0 else 'left'] * abs(dist))
        else:
            actions.extend(['down' if dist > 0 else 'up'] * abs(dist))
    return ' '.join(actions), skipped


def check_path(world, actions_str):
    """
    Simulate the action sequence on the world grid.
    Returns None on success, or a plain-English error string.
    """
    start = goal = None
    for r, row in enumerate(world):
        for c, val in enumerate(row):
            if val == 2:
                start = (r, c)
            elif val == 3:
                goal = (r, c)

    actions = actions_str.split() if actions_str else []
    if not actions:
        return "No valid move_x/move_y calls parsed from response"

    deltas = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
    pos = start
    for i, action in enumerate(actions):
        dr, dc = deltas[action]
        r, c = pos[0] + dr, pos[1] + dc
        if r < 0 or r >= len(world) or c < 0 or c >= len(world[0]):
            return f"Out of bounds at step {i + 1}: tried to move to ({r}, {c})"
        if world[r][c] == 1:
            return f"Hit obstacle at step {i + 1}: ({r}, {c})"
        pos = (r, c)

    if pos != goal:
        return f"Path ended at {pos} but goal is at {goal}"
    return None


def _resolve_attempt(r):
    """
    Return (response_str, error_str_or_None, successful_try_or_None)
    for a single record, handling both legacy and multi-try formats.
    """
    attempts = r.get('attempts')

    if attempts:
        # Multi-try format: pick first successful attempt, or last if all failed.
        for a in attempts:
            if a['error'] is None:
                return a['response'], None, a['try']
        # All failed — use last attempt (error is already stored).
        last = attempts[-1]
        return last['response'], last['error'], None

    else:
        # Legacy single-response format.
        response = r['response']
        actions, skipped = parse_response(response)
        if skipped:
            error = f"Non-integer argument(s) in: {', '.join(skipped)}"
        else:
            error = check_path(r['world'], actions)
        successful_try = 1 if error is None else None
        return response, error, successful_try


def main():
    if len(sys.argv) != 3:
        print("Usage: python parse_code_form.py <input.jsonl> <output.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        records = [json.loads(line) for line in f]

    out = []
    for r in records:
        response, error, successful_try = _resolve_attempt(r)
        actions, _ = parse_response(response)

        out.append({
            "english":        r['nl_description'],
            "ground_truth":   r['ground_truth'],
            "generated":      [actions],
            "world":          r['world'],
            "error":          error,
            "successful_try": successful_try,
        })

    with open(sys.argv[2], 'w') as f:
        json.dump(out, f, indent=2)

    n_errors = sum(1 for r in out if r['error'])
    print(f"Parsed {len(out)} samples ({n_errors} errors) -> {sys.argv[2]}")

    # Per-try success breakdown (only meaningful for multi-try runs).
    max_try = max(
        (max(a['try'] for a in r['attempts']) for r in records if r.get('attempts')),
        default=1,
    )
    if max_try > 1:
        print("Success breakdown by try:")
        for t in range(1, max_try + 1):
            n = sum(1 for r in out if r['successful_try'] == t)
            print(f"  Try {t}: {n}")
        print(f"  Failed all: {sum(1 for r in out if r['successful_try'] is None)}")


if __name__ == '__main__':
    main()

"""
Convert text_feedback.py JSONL output into the format expected by executor-point-sg.py.

Direction tokens (up, down, left, right) are read directly from each attempt's
response; no expansion step is needed (unlike parse_code_form.py).

Supports both output formats:
  - Multi-try (text_feedback.py): record has an "attempts" list; the first
    attempt whose "error" is null is used. If all failed, the last is used.
  - Single-response (run_baseline.py): record has a top-level "response" field.

Each output record includes:
  "error":          null on success, or a plain-English error description.
  "successful_try": 1-based try number that succeeded, or null if all failed.
"""
import json
import sys


def parse_response(response: str) -> str:
    """Extract direction tokens from a free-form response string."""
    return ' '.join(t for t in response.split() if t in ('up', 'down', 'left', 'right'))


def check_path(world, actions_str: str):
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
        return "No valid action tokens parsed from response"

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
    for a single record, handling both multi-try and single-response formats.
    """
    attempts = r.get('attempts')

    if attempts:
        # Multi-try format: pick first successful attempt, or last if all failed.
        for a in attempts:
            if a['error'] is None:
                return a['response'], None, a['try']
        last = attempts[-1]
        return last['response'], last['error'], None

    else:
        # Single-response format (run_baseline.py output).
        response = r['response']
        actions  = parse_response(response)
        error    = check_path(r['world'], actions)
        return response, error, (1 if error is None else None)


def main():
    if len(sys.argv) != 3:
        print("Usage: python parse_text_feedback.py <input.jsonl> <output.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Separate config line (if present) from sample records.
    records = [r for r in records if not r.get('_config')]

    out = []
    for r in records:
        response, error, successful_try = _resolve_attempt(r)
        actions = parse_response(response)

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

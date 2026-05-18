"""
Convert code_form.py JSONL output into the format expected by executor-point-sg.py.

Each move_x(d) / move_y(d) call is expanded into d repeated action tokens:
  move_x(+d) -> "right" * d
  move_x(-d) -> "left"  * d
  move_y(+d) -> "down"  * d
  move_y(-d) -> "up"    * d

Calls with non-integer arguments (e.g. variables) are skipped, which will
cause evaluation to fail — intentionally surfacing LLM mistakes.
"""
import json
import re
import sys


def parse_response(response):
    actions = []
    for axis, dist_str in re.findall(r'move_([xy])\((-?\d+)\)', response):
        dist = int(dist_str)
        if axis == 'x':
            actions.extend(['right' if dist > 0 else 'left'] * abs(dist))
        else:
            actions.extend(['down' if dist > 0 else 'up'] * abs(dist))
    return ' '.join(actions)


def main():
    if len(sys.argv) != 3:
        print("Usage: python parse_code_form.py <input.jsonl> <output.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        records = [json.loads(line) for line in f]

    out = [
        {
            "english":      r['nl_description'],
            "ground_truth": r['ground_truth'],
            "generated":    [parse_response(r['response'])],
            "world":        r['world'],
        }
        for r in records
    ]

    with open(sys.argv[2], 'w') as f:
        json.dump(out, f, indent=2)

    print(f"Parsed {len(out)} samples -> {sys.argv[2]}")


if __name__ == '__main__':
    main()

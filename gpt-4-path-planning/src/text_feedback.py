import os
import json
import time
import argparse
from openai import OpenAI
from dotenv import load_dotenv
from vllm_utils import launch_vllm_server, strip_thinking, vllm_extra_body

load_dotenv()

# ---------------------------------------------------------------------------
# One-shot example — same navigation scenario used in code_form.py.
# World: 25x25, rectangular obstacle block at rows 5-9 cols 3-7,
#        start (2,5), goal (12,5).
# Solution path: down 2, right 5, down 8, left 5  (same route as code_form)
# ---------------------------------------------------------------------------

# 'text' / naive_representation format
ONE_SHOT_INPUT_TEXT = (
    "You are in a 25 by 25 world. "
    "There are obstacles that you have to avoid at: "
    "(5,3), (5,4), (5,5), (5,6), (5,7), "
    "(6,3), (6,4), (6,5), (6,6), (6,7), "
    "(7,3), (7,4), (7,5), (7,6), (7,7), "
    "(8,3), (8,4), (8,5), (8,6), (8,7), "
    "(9,3), (9,4), (9,5), (9,6), (9,7). "
    "Go from (2,5) to (12,5)"
)

# 'code' / code_representation format
ONE_SHOT_INPUT_CODE = """\

#The goal is to navigate a 25x25 grid to go from the initial location to the goal while avoiding obstacles

obstacles = []
goals = [(12, 5)]
initial_location = (2, 5)

for i in range(5, 9):
    for j in range(3, 7):
        obstacles.append((i, j))
        """

# 'grid' / grid_representation format
ONE_SHOT_INPUT_GRID = (
    "0000000000000000000000000\n"
    "0000000000000000000000000\n"
    "0000020000000000000000000\n"   # row  2: start at col 5
    "0000000000000000000000000\n"
    "0000000000000000000000000\n"
    "0001111100000000000000000\n"   # rows 5-9: obstacle block, cols 3-7
    "0001111100000000000000000\n"
    "0001111100000000000000000\n"
    "0001111100000000000000000\n"
    "0001111100000000000000000\n"
    "0000000000000000000000000\n"
    "0000000000000000000000000\n"
    "0000030000000000000000000\n"   # row 12: goal at col 5
    "0000000000000000000000000\n"
    "0000000000000000000000000\n"
    "0000000000000000000000000\n"
    "0000000000000000000000000\n"
    "0000000000000000000000000\n"
    "0000000000000000000000000\n"
    "0000000000000000000000000\n"
    "0000000000000000000000000\n"
    "0000000000000000000000000\n"
    "0000000000000000000000000\n"
    "0000000000000000000000000\n"
    "0000000000000000000000000\n"
)

# The direction-token solution is identical regardless of input representation.
# Route: down 2 (row 2→4), right 5 (col 5→10), down 8 (row 4→12), left 5 (col 10→5)
ONE_SHOT_ACTIONS = (
    "down down "                                   # row 2 -> row 4
    "right right right right right "               # col 5 -> col 10  (clears obstacle cols 3-7)
    "down down down down down down down down "      # row 4 -> row 12  (clears obstacle rows 5-9)
    "left left left left left"                     # col 10 -> col 5 = goal
)

# Per-representation metadata used in build_prompt.
_REPR_CONFIG = {
    'text': {
        'one_shot_input': ONE_SHOT_INPUT_TEXT,
        'label':          'Task',
        'inline':         True,
        'sample_field':   'naive_representation',
    },
    'code': {
        'one_shot_input': ONE_SHOT_INPUT_CODE,
        'label':          'Description in code',
        'inline':         False,
        'sample_field':   'code_representation',
    },
    'grid': {
        'one_shot_input': ONE_SHOT_INPUT_GRID,
        'label':          'Grid representation',
        'inline':         False,
        'sample_field':   'grid_representation',
    },
}


def _format_task(label: str, inline: bool, task_input: str) -> str:
    if inline:
        return f"{label}: {task_input}\n"
    else:
        return f"{label}:\n{task_input}\n"


def build_prompt(task_input: str, representation: str = 'text') -> str:
    """
    Build the text-feedback one-shot prompt.

    Coordinate convention (matching the task descriptions):
      - Positions are given as (row, col).
      - The top-left corner is (0, 0); rows increase downward, cols rightward.

    The expected output is a space-separated sequence of direction tokens:
      up / down / left / right
    one token per step.
    """
    cfg = _REPR_CONFIG[representation]
    example_task = _format_task(cfg['label'], cfg['inline'], cfg['one_shot_input'])
    query_task   = _format_task(cfg['label'], cfg['inline'], task_input)

    return (
        "Provide a sequence of direction tokens (up, down, left, right), one per step, "
        "to navigate from start to goal without hitting obstacles.\n"
        "Positions are given as (row, col). The top-left corner is (0, 0); rows increase "
        "downward, columns increase rightward.\n"
        "Output only the space-separated token sequence, nothing else.\n"
        "\n"
        "For example:\n"
        f"{example_task}"
        f"Actions: {ONE_SHOT_ACTIONS}\n"
        "\n"
        "Now solve this maze:\n"
        f"{query_task}"
        "Actions: "
    )


def build_feedback_message(error: str) -> str:
    """Feedback message appended to the conversation when a solution is wrong.
    Structure mirrors code_form.py's build_feedback_message."""
    return (
        f"That solution is incorrect.\n"
        f"Error: {error}\n"
        "Please write a corrected action sequence.\n"
        "Actions: "
    )


def parse_response(response: str) -> str:
    """Extract direction tokens from a free-form response string."""
    return ' '.join(t for t in response.split() if t in ('up', 'down', 'left', 'right'))


def check_path(world, actions_str: str):
    """
    Simulate the action sequence on the world grid.
    Returns None on success, or a plain-English error string.
    Error messages are identical to those in code_form.py / parse_code_form.py.
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


def call_api(client: OpenAI, model: str, messages: list) -> str:
    """Call the API with one retry on failure."""
    extra_body = vllm_extra_body(model)
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=600,
                extra_body=extra_body,
            )
            return strip_thinking(response.choices[0].message.content)
        except Exception as e:
            if attempt == 0:
                print(f"API error: {e}. Retrying in 25s...")
                time.sleep(25)
            else:
                raise


def main():
    """
    Usage:
        python text_feedback.py <test_json> <output_jsonl>
                                [--representation {text,code,grid}]
                                [--max-samples N]
                                [--max-tries N]

    Ablation counterpart to code_form.py: same retry/feedback loop but the
    LLM is asked for direction tokens (up/down/left/right) instead of a
    Python solve() function.  All other parameters match code_form.py.

    Each output record (one JSON line) contains:
        { "id": int, "naive_representation": str, "ground_truth": str, "world": list,
          "attempts": [{"try": int, "prompt": str, "response": str,
                        "error": str|null}, ...] }
    """
    parser = argparse.ArgumentParser(
        description="Text-output with feedback loop: ablation for gpt-4-path-planning datasets."
    )
    parser.add_argument("test_json",   help="Input JSON file (list of samples).")
    parser.add_argument("output_jsonl", help="Output JSONL file.")
    parser.add_argument(
        "--representation", choices=["text", "code", "grid"], default="text",
        help="Input representation to use in the prompt (default: text).",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit number of samples processed (default: all).",
    )
    parser.add_argument(
        "--max-tries", type=int, default=1,
        help="Max inference rounds per sample with error feedback (default: 1).",
    )
    parser.add_argument(
        "--provider", choices=["openai", "vllm", "stanford"], default="openai",
        help="Model provider (default: openai).",
    )
    parser.add_argument(
        "--model", default="gpt-4",
        help="Model name. Default: %(default)s.",
    )
    parser.add_argument(
        "--base-url", default="http://localhost:8000/v1",
        help="Base URL for the vLLM OpenAI-compatible endpoint (only used with "
             "--provider vllm). Default: %(default)s.",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="Number of GPUs for tensor parallelism when launching vLLM. Default: 1.",
    )
    parser.add_argument(
        "--launch-vllm", action="store_true",
        help="Spawn a vLLM server subprocess before running inference.",
    )
    args = parser.parse_args()

    if args.provider == "vllm":
        if args.launch_vllm:
            launch_vllm_server(args.model, args.base_url, args.tensor_parallel_size)
        client = OpenAI(api_key="EMPTY", base_url=args.base_url)
    elif args.provider == "stanford":
        client = OpenAI(
            api_key=os.environ.get("STANFORD_API_KEY"),
            base_url="https://aiapi-prod.stanford.edu/v1",
        )
    else:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    model          = args.model
    representation = args.representation
    sample_field   = _REPR_CONFIG[representation]['sample_field']

    data = json.load(open(args.test_json))
    if args.max_samples is not None:
        data = data[:args.max_samples]

    print(f"Running: provider={args.provider}, model={model}, "
          f"representation={representation}, max_tries={args.max_tries}, "
          f"samples={len(data)}.")

    config = {
        "_config":             True,
        "provider":            args.provider,
        "model":               model,
        "tensor_parallel_size": args.tensor_parallel_size if args.provider == "vllm" else None,
        "temperature":         0.0,
        "representation":      representation,
        "argv":                vars(args),
    }

    # Resume: load any already-completed records from the output file.
    results = []
    completed_ids = set()
    if os.path.exists(args.output_jsonl):
        with open(args.output_jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("_config"):
                    continue
                results.append(rec)
                completed_ids.add(rec["id"])
        if completed_ids:
            print(f"Resuming: loaded {len(completed_ids)} completed sample(s) "
                  f"(ids {min(completed_ids)}–{max(completed_ids)}).")

    for i, sample in enumerate(data):
        if i in completed_ids:
            print(f"\n--- Sample {i} (already done, skipping) ---")
            continue
        print(f"\n--- Sample {i} ---")
        initial_prompt = build_prompt(sample[sample_field], representation)

        messages = [{"role": "user", "content": initial_prompt}]
        attempts = []

        for try_num in range(1, args.max_tries + 1):
            logged_prompt = messages[-1]["content"]
            print(f"  [Try {try_num}] Prompt:\n{logged_prompt}")

            raw = call_api(client, model, messages)
            print(f"  [Try {try_num}] Response:\n{raw}")

            actions = parse_response(raw)
            error = check_path(sample['world'], actions)

            attempts.append({
                "try":      try_num,
                "prompt":   logged_prompt,
                "response": raw,
                "error":    error,
            })

            if error is None:
                print(f"  [Try {try_num}] Success!")
                break

            print(f"  [Try {try_num}] Error: {error}")
            if try_num < args.max_tries:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user",
                                  "content": build_feedback_message(error)})

        result = {
            "id":                   i,
            "naive_representation": sample['naive_representation'],
            "ground_truth":         sample['path'],
            "world":                sample['world'],
            "attempts":             attempts,
        }
        results.append(result)

        # Write after each sample so progress is not lost on interruption.
        with open(args.output_jsonl, 'w') as f:
            f.write(json.dumps(config) + '\n')
            for r in results:
                f.write(json.dumps(r) + '\n')

    print(f"\nDone. Saved {len(results)} samples to {args.output_jsonl}")


if __name__ == '__main__':
    main()

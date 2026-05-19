import os
import json
import time
import argparse
from openai import OpenAI
from dotenv import load_dotenv
from parse_code_form import parse_response, check_path
from vllm_utils import launch_vllm_server, strip_thinking, vllm_extra_body

load_dotenv()

# ---------------------------------------------------------------------------
# One-shot example — same navigation scenario presented in three formats.
# World: 25x25, rectangular obstacle block at rows 5-9 cols 3-7,
#        start (2,5), goal (12,5).
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

# 'code' / code_representation format  (matches generate_code_rectangle style)
ONE_SHOT_INPUT_CODE = """\

#The goal is to navigate a 25x25 grid to go from the initial location to the goal while avoiding obstacles

obstacles = []
goals = [(12, 5)]
initial_location = (2, 5)

for i in range(5, 9):
    for j in range(3, 7):
        obstacles.append((i, j))
        """

# 'grid' / grid_representation format  (matches generate_grid style)
# 25 rows x 25 cols; 2=start, 3=goal, 1=obstacle, 0=free
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

# The solve() output is identical regardless of input representation.
ONE_SHOT_SOLVE = """\
def solve():
    # start (2,5), goal (12,5); rectangular block at rows 5-9, cols 3-7 blocks direct descent
    move_y(2)   # down 2 rows:  row 2  -> row 4   (stop just before the obstacle block)
    move_x(5)   # right 5 cols: col 5  -> col 10  (step right of the block which ends at col 7)
    move_y(8)   # down 8 rows:  row 4  -> row 12  (descend past the block which ends at row 9)
    move_x(-5)  # left 5 cols:  col 10 -> col 5   (back to goal column; now at row 12, col 5 = goal)"""

# Per-representation metadata used in build_prompt.
_REPR_CONFIG = {
    'text': {
        'one_shot_input': ONE_SHOT_INPUT_TEXT,
        'label':          'Task',          # "Task: <input>"
        'inline':         True,            # input follows label on same line
        'sample_field':   'naive_representation',
    },
    'code': {
        'one_shot_input': ONE_SHOT_INPUT_CODE,
        'label':          'Description in code',  # "Description in code:\n<input>"
        'inline':         False,
        'sample_field':   'code_representation',
    },
    'grid': {
        'one_shot_input': ONE_SHOT_INPUT_GRID,
        'label':          'Grid representation',  # "Grid representation:\n<input>"
        'inline':         False,
        'sample_field':   'grid_representation',
    },
}


def _format_task(label: str, inline: bool, task_input: str) -> str:
    """Return 'Label: input\n' or 'Label:\ninput\n' depending on inline."""
    if inline:
        return f"{label}: {task_input}\n"
    else:
        return f"{label}:\n{task_input}\n"


def build_prompt(task_input: str, representation: str = 'text') -> str:
    """
    Build the code-form one-shot prompt.

    Coordinate convention (matching the task descriptions):
      - Positions are given as (row, col).
      - The top-left corner is (0, 0); rows increase downward, cols increase rightward.
      - move_x(d): move d steps along columns  (+d = right, -d = left)
      - move_y(d): move d steps along rows     (+d = down,  -d = up)
    """
    cfg = _REPR_CONFIG[representation]
    example_task = _format_task(cfg['label'], cfg['inline'], cfg['one_shot_input'])
    query_task   = _format_task(cfg['label'], cfg['inline'], task_input)

    return (
        "Use the functions move_x(dist:int) and move_y(dist:int) to write a sequence of "
        "steps as a Python function that will successfully navigate the maze. "
        "You can use the variables N (height) and M (width), but the functions move_x and "
        "move_y will automatically bound your movements to the allowable area.\n"
        "Positions are given as (row, col). The top-left corner is (0, 0); rows increase "
        "downward, columns increase rightward. "
        "move_x(d) moves d steps rightward (+d) or leftward (-d). "
        "move_y(d) moves d steps downward (+d) or upward (-d).\n"
        "Do NOT implement a graph search algorithm; just write the steps for this specific "
        "maze. Enclose your code in a function solve().\n"
        "\n"
        "For example:\n"
        f"{example_task}"
        "```python\n"
        f"{ONE_SHOT_SOLVE}\n"
        "```\n"
        "\n"
        "Now write a function to solve this maze:\n"
        f"{query_task}"
        "```python\n"
    )


def build_feedback_message(error: str) -> str:
    """Feedback message appended to the conversation when a solution is wrong."""
    return (
        f"That solution is incorrect.\n"
        f"Error: {error}\n"
        "Please write a corrected solve() function.\n"
        "```python\n"
    )


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
        python code_form.py <test_json> <output_jsonl>
                            [--representation {text,code,grid}]
                            [--max-samples N]
                            [--max-tries N]

    --representation selects the input format fed to the model:
        text  (default) naive_representation field — plain-English obstacle list
        code            code_representation field  — Python obstacle-loop snippet
        grid            grid_representation field  — 2-D digit grid

    --max-tries controls how many inference rounds per sample (default 1).
    On each failed attempt the LLM receives its previous response plus a
    plain-English error description and is asked to correct the solution.

    Each output record (one JSON line) contains:
        { "id": int, "naive_representation": str, "ground_truth": str, "world": list,
          "attempts": [{"try": int, "prompt": str, "response": str,
                        "error": str|null}, ...] }

    "prompt" for try 1 is the full initial prompt.
    "prompt" for tries 2+ is just the feedback message added in that turn
    (the full conversation is reconstructable from all attempts).
    """
    parser = argparse.ArgumentParser(
        description="Code-form path-planning inference for gpt-4-path-planning datasets."
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
        "--provider", choices=["openai", "vllm"], default="openai",
        help="Model provider (default: openai).",
    )
    parser.add_argument(
        "--model", default="gpt-4",
        help="Model name. For --provider vllm, use the HuggingFace model name "
             "(e.g. deepseek-ai/deepseek-moe-16b-chat). Default: %(default)s.",
    )
    parser.add_argument(
        "--base-url", default="http://localhost:8000/v1",
        help="Base URL for the vLLM OpenAI-compatible endpoint (only used with "
             "--provider vllm). Default: %(default)s.",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="Number of GPUs for tensor parallelism when launching vLLM "
             "(only used with --provider vllm --launch-vllm). Default: 1.",
    )
    parser.add_argument(
        "--launch-vllm", action="store_true",
        help="Spawn a vLLM server subprocess before running inference. "
             "The server is terminated automatically when the script exits.",
    )
    args = parser.parse_args()

    if args.provider == "vllm":
        if args.launch_vllm:
            launch_vllm_server(args.model, args.base_url, args.tensor_parallel_size)
        client = OpenAI(api_key="EMPTY", base_url=args.base_url)
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

    results = []
    for i, sample in enumerate(data):
        print(f"\n--- Sample {i} ---")
        initial_prompt = build_prompt(sample[sample_field], representation)

        # Build the message history for this sample; grows on retries.
        messages = [{"role": "user", "content": initial_prompt}]
        attempts = []

        for try_num in range(1, args.max_tries + 1):
            logged_prompt = messages[-1]["content"]
            print(f"  [Try {try_num}] Prompt:\n{logged_prompt}")

            raw = call_api(client, model, messages)
            print(f"  [Try {try_num}] Response:\n{raw}")

            # Parse inline so we can provide error feedback on the next turn.
            actions, skipped = parse_response(raw)
            if skipped:
                error = f"Non-integer argument(s) in: {', '.join(skipped)}"
            else:
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
                # Extend conversation: assistant sends its (wrong) code,
                # user sends the error feedback.
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
        # Config line is always first so the file is self-identifying.
        with open(args.output_jsonl, 'w') as f:
            f.write(json.dumps(config) + '\n')
            for r in results:
                f.write(json.dumps(r) + '\n')

    print(f"\nDone. Saved {len(results)} samples to {args.output_jsonl}")


if __name__ == '__main__':
    main()

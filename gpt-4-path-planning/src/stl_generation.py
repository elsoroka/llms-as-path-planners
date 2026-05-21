import os
import sys
import json
import time
import argparse
from openai import OpenAI
from dotenv import load_dotenv
from vllm_utils import launch_vllm_server, strip_thinking, vllm_extra_body

# Import check_stl from the PPNL ICL directory.
_PPNL_ICL = os.path.join(os.path.dirname(__file__), '..', '..', 'ppnl-spatial-temporal-reasoning', 'ICL')
sys.path.insert(0, os.path.abspath(_PPNL_ICL))
from parse_stl import check_stl

load_dotenv()

# ---------------------------------------------------------------------------
# One-shot example — same navigation scenario as code_form.py.
# World: 25x25, obstacle block range(5,9) x range(3,7) = rows 5-8, cols 3-6.
# Start (2,5), goal (12,5).
# ---------------------------------------------------------------------------

ONE_SHOT_INPUT_CODE = """\

#The goal is to navigate a 25x25 grid to go from the initial location to the goal while avoiding obstacles

obstacles = []
goals = [(12, 5)]
initial_location = (2, 5)

for i in range(5, 9):
    for j in range(3, 7):
        obstacles.append((i, j))
        """

STL_PROMPT = """Your task is to write a Python function safe_paths(x, y) that returns a temporal logic Predicate defining safe paths to the goal.

x and y are Trajectory objects — sequences of coordinate values along a path. They are already provided; do not redefine them and do not redefine any of the operators below.

You may build Predicates directly from Trajectories using: ==, <, >, <=, >=
  e.g.  x == 3      (True at every step where the x-coordinate is 3)
        y < 2       (True at every step where the y-coordinate is less than 2)

Combine Predicates with the Boolean operators:
* And(p, q, ...) -> Predicate   (all must hold at each step)
* Or(p, q, ...)  -> Predicate   (at least one holds at each step)
* Not(p)         -> Predicate   (negation at each step)
* Implies(p, q)  -> Predicate

Wrap Predicates in temporal operators:
* Always(p)      -> Predicate   (p holds at every step)
* Eventually(p)  -> Predicate   (p holds at some step)
* Until(p, q)    -> Predicate   (p holds until q becomes true)

Do NOT import any modules, redefine any of these operators, or use lambda functions.
Your function must return a single Predicate built from the operators above.
"""

ONE_SHOT_CODE = """\
def safe_paths(x, y):
    in_obstacle = And(x >= 5, x <= 8, y >= 3, y <= 6)
    obstacle_avoidance = Always(Not(in_obstacle))
    reach_goal = Eventually(And(x == 12, y == 5))
    return And(obstacle_avoidance, reach_goal)"""


def reconstruct_path(start, path_str):
    """Reconstruct solution_coordinates from start and space-separated direction tokens."""
    dirs = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
    coords = [list(start)]
    for token in path_str.strip().split():
        dr, dc = dirs[token]
        coords.append([coords[-1][0] + dr, coords[-1][1] + dc])
    return coords


def build_prompt(sample: dict) -> str:
    """Build the STL generation one-shot prompt for a single gpt-4 task."""
    return (
        STL_PROMPT + "\n"
        "For example:\n"
        "Description in code:\n"
        f"{ONE_SHOT_INPUT_CODE}\n"
        "```python\n"
        f"{ONE_SHOT_CODE}\n"
        "```\n"
        "\n"
        "Now write a temporal logic predicate for this task:\n"
        "Description in code:\n"
        f"{sample['code_representation']}\n"
        "```python\n"
    )


def build_feedback_message(error: str) -> str:
    return (
        f"That solution is incorrect.\n"
        f"Error: {error}\n"
        "Please write a corrected safe_paths() function.\n"
        "```python\n"
    )


def call_api(client: OpenAI, model: str, messages: list) -> str:
    extra_body = vllm_extra_body(model)
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
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
    Each output record (one JSON line) contains:
        { "id": int, "code_representation": str, "ground_truth": str, "world": list,
          "attempts": [{"try": int, "prompt": str, "response": str,
                        "error": str|null}, ...] }
    """
    parser = argparse.ArgumentParser(
        description="STL code-form generation for gpt-4-path-planning datasets."
    )
    parser.add_argument("test_json",    help="Input JSON file (list of samples).")
    parser.add_argument("output_jsonl", help="Output JSONL file.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tries",   type=int, default=1)
    parser.add_argument("--provider", choices=["openai", "vllm", "stanford"], default="openai")
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--launch-vllm", action="store_true")
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

    model       = args.model
    max_tries   = args.max_tries
    output_file = args.output_jsonl

    data = json.load(open(args.test_json))
    if args.max_samples is not None:
        data = data[:args.max_samples]

    print(f"Running: provider={args.provider}, model={model}, "
          f"max_tries={max_tries}, samples={len(data)}.")

    config = {
        "_config":              True,
        "provider":             args.provider,
        "model":                model,
        "tensor_parallel_size": args.tensor_parallel_size if args.provider == "vllm" else None,
        "temperature":          0.0,
        "argv":                 vars(args),
    }

    results = []
    done_ids = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("_config"):
                    continue
                # Only keep/skip samples that already succeeded; re-run failures.
                if rec.get("attempts") and rec["attempts"][-1]["error"] is None:
                    results.append(rec)
                    done_ids.add(rec["id"])
        if done_ids:
            print(f"Resuming: {len(done_ids)} succeeded samples skipped, re-running failures.")

    for i, sample in enumerate(data):
        if i in done_ids:
            print(f"\n--- Sample {i} (skipped, already done) ---")
            continue
        print(f"\n--- Sample {i} ---")

        stl_sample = {
            "world":                sample["world"],
            "solution_coordinates": reconstruct_path(sample["start"], sample["path"]),
        }

        initial_prompt = build_prompt(sample)
        messages = [{"role": "user", "content": initial_prompt}]
        attempts = []

        for try_num in range(1, max_tries + 1):
            logged_prompt = messages[-1]["content"]
            print(f"  [Try {try_num}] Prompt:\n{logged_prompt}")

            raw = call_api(client, model, messages)
            print(f"  [Try {try_num}] Response:\n{raw}")

            start = raw.find("```python")
            if start == -1:
                start = raw.find("```")
            if start == -1:
                error = "No ```python block found in response"
                attempts.append({"try": try_num, "prompt": logged_prompt,
                                  "response": raw, "error": error})
                break
            start = raw.find("\n", start) + 1
            end = raw.find("```", start)
            stl = raw[start:end].strip() if end != -1 else raw[start:].strip()
            error = check_stl(stl_sample, stl)

            attempts.append({"try": try_num, "prompt": logged_prompt,
                              "response": raw, "error": error})

            if error is None:
                print(f"  [Try {try_num}] Success!")
                break

            print(f"  [Try {try_num}] Error: {error}")
            if try_num < max_tries:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": build_feedback_message(error)})

        result = {
            "id":                  i,
            "code_representation": sample["code_representation"],
            "ground_truth":        sample["path"],
            "world":               sample["world"],
            "attempts":            attempts,
        }
        results.append(result)

        with open(output_file, 'w') as f:
            f.write(json.dumps(config) + '\n')
            for r in results:
                f.write(json.dumps(r) + '\n')

    print(f"\nDone. Saved {len(results)} samples to {output_file}")


if __name__ == '__main__':
    main()

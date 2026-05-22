import argparse
import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
from parse_stl import check_stl
from vllm_utils import launch_vllm_server, strip_thinking, vllm_extra_body

load_dotenv()

# One-shot example task and solution (same environment used in the 5-shot baseline prompt)
ONE_SHOT_TASK = (
    "You are in a 6 by 6 world. "
    "There are obstacles that you have to avoid at: (2,1). "
    "You have to reach the goal (0,1). "
)
STL_PROMPT = """Your task is to write a Python function safe_paths(x, y) that returns a temporal logic Predicate defining safe paths to the goal.

x and y are Trajectory objects — sequences of coordinate values along a path. They are already provided; do not redefine them and do not redefine Any of the operators below.

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
* Eventually(p)  -> Predicate   (p holds at some step).
* Until(p, q)    -> Predicate   (p holds until q becomes true)

Do NOT import any modules, redefine any of these operators, or use lambda functions.
Your function must return a single Predicate built from the operators above.
"""

ONE_SHOT_CODE = """\
def safe_paths(x, y):
    in_obstacle = And(x == 2, y == 1)
    obstacle_avoidance = Always(Not(in_obstacle))
    reach_goal = Eventually(And(x == 0, y == 1))
    return And(obstacle_avoidance, reach_goal)"""

def build_prompt(sample: dict) -> str:
    """
    Build the STL generation one-shot prompt for a single maze task.
    """
    stem = sample['nl_description'].split("Go from")[0].strip()
    goal = f"You have to reach the goal ({sample['solution_coordinates'][-1][0]}, {sample['solution_coordinates'][-1][1]})."

    return (
        STL_PROMPT + "\n"
        "For example:\n"
        f"Task: {ONE_SHOT_TASK}\n"
        "```python\n"
        f"{ONE_SHOT_CODE}\n"
        "```\n"
        "\n"
        "Now write a temporal logic predicate for this task:\n"
        f"Task: {stem}\n{goal}\n"
        "```python\n"
    )


def build_feedback_message(error: str) -> str:
    """Feedback message appended to the conversation when a solution is wrong."""
    return (
        f"That solution is incorrect.\n"
        f"Error: {error}\n"
        "Please write a corrected safe_paths() function.\n"
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
        { "id": int, "nl_description": str, "ground_truth": str, "world": list,
          "attempts": [{"try": int, "prompt": str, "response": str,
                        "error": str|null}, ...] }

    "prompt" for try 1 is the full initial prompt.
    "prompt" for tries 2+ is just the feedback message added in that turn
    (the full conversation is reconstructable from all attempts).
    """
    parser = argparse.ArgumentParser(
        description="STL generation for PPNL single-goal benchmark."
    )
    parser.add_argument("test_json",    help="Input JSON file (list of samples).")
    parser.add_argument("output_jsonl", help="Output JSONL file.")
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
        "--model", default="gpt-4.1",
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

    print(f"Running with provider={args.provider}, model={model}, "
          f"max_tries={max_tries} on {len(data)} samples.")

    config = {
        "_config":             True,
        "provider":            args.provider,
        "model":               model,
        "tensor_parallel_size": args.tensor_parallel_size if args.provider == "vllm" else None,
        "temperature":         0.0,
        "argv":                vars(args),
    }

    # Resume: load any results already written to the output file.
    results = []
    done_ids = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("_config"):
                    continue
                if rec.get("attempts") and rec["attempts"][-1]["error"] is None or '```' not in rec["attempts"][-1]["error"]:
                    results.append(rec)
                    done_ids.add(rec["id"])
        if done_ids:
            print(f"Resuming: {len(done_ids)} succeeded samples skipped, re-running failures.")

    for i, sample in enumerate(data):
        if i in done_ids:
            print(f"\n--- Sample {i} (skipped, already done) ---")
            continue
        print(f"\n--- Sample {i} ---")
        sc = sample.get('solution_coordinates')
        if not isinstance(sc, list) or not sc or not isinstance(sc[-1], (list, tuple)) or len(sc[-1]) < 2:
            print("  Skipping: no valid solution_coordinates")
            continue
        initial_prompt = build_prompt(sample)

        # Build the message history for this sample; grows on retries.
        messages = [{"role": "user", "content": initial_prompt}]
        attempts = []

        for try_num in range(1, max_tries + 1):
            logged_prompt = messages[-1]["content"]
            print(f"  [Try {try_num}] Prompt:\n{logged_prompt}")

            raw = call_api(client, model, messages)
            print(f"  [Try {try_num}] Response:\n{raw}")

            # Parse inline so we can provide error feedback on the next turn.
            start = raw.find("```python")
            if start == -1:
                start = raw.find("```")
            if start == -1:
                stl = raw.strip()
            else:
                start = raw.find("\n", start) + 1  # skip past the opening fence line
                end = raw.find("```", start)
                stl = raw[start:end].strip() if end != -1 else raw[start:].strip()
            error = check_stl(sample, stl)

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
            if try_num < max_tries:
                # Extend conversation: assistant sends its (wrong) code,
                # user sends the error feedback.
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user",
                                  "content": build_feedback_message(error)})

        result = {
            "id":             i,
            "nl_description": sample['nl_description'],
            "ground_truth":   sample['agent_as_a_point'],
            "world":          sample['world'],
            "attempts":       attempts,
        }
        results.append(result)

        # Write after each sample so progress is not lost on interruption.
        # Config line is always first so the file is self-identifying.
        with open(output_file, 'w') as f:
            f.write(json.dumps(config) + '\n')
            for r in results:
                f.write(json.dumps(r) + '\n')

    print(f"\nDone. Saved {len(results)} samples to {output_file}")


if __name__ == '__main__':
    main()

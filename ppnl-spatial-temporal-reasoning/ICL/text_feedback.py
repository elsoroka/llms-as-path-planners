"""
Text-output with feedback loop: ablation for PPNL single-goal benchmark.

Uses the same 5-shot ICL prompt as run_baseline.py but adds the same
error-feedback retry loop as code_form.py.  Output format is identical to
code_form.py (attempts list) so the same evaluation pipeline applies.

Usage:
    python text_feedback.py <test_json> <output_jsonl>
                            [--max-samples N]
                            [--max-tries N]

Each output record (one JSON line) contains:
    { "id": int, "nl_description": str, "ground_truth": str, "world": list,
      "attempts": [{"try": int, "prompt": str, "response": str,
                    "error": str|null}, ...] }

"prompt" for try 1 is the full initial prompt.
"prompt" for tries 2+ is just the feedback message added in that turn
(the full conversation is reconstructable from all attempts).
"""

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from vllm_utils import launch_vllm_server, strip_thinking, vllm_extra_body

load_dotenv()

PROMPT_FILE   = Path(__file__).parent / 'prompts' / 'few-shot-prompts-single_goal_5.txt'
DEFAULT_MODEL = 'gpt-4'
TEMPERATURE   = 0.0
MAX_TOKENS    = 1024


def load_prompt_template() -> str:
    return PROMPT_FILE.read_text().rstrip()


def build_initial_prompt(template: str, nl_description: str) -> str:
    """Build the full first-turn prompt: 5-shot template + query task."""
    return f"""{template}
###
Task: {nl_description}
Actions: """


def parse_response(response: str) -> str:
    """Extract direction tokens from a free-form response string."""
    return ' '.join(t for t in response.split() if t in ('up', 'down', 'left', 'right'))


def check_path(world, actions_str: str):
    """
    Simulate the action sequence on the world grid.
    Returns None on success, or a plain-English error string.
    Error messages are intentionally identical to those in code_form.py /
    parse_code_form.py so the two ablations are directly comparable.
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


def build_feedback_message(error: str) -> str:
    """Feedback message appended to the conversation when a solution is wrong.
    Structure mirrors code_form.py's build_feedback_message so the two
    ablations receive the same information."""
    return (
        f"That solution is incorrect.\n"
        f"Error: {error}\n"
        "Please write a corrected action sequence.\n"
        "Actions: "
    )


def call_api(client: OpenAI, model: str, messages: list) -> str:
    """Call the API with one retry on failure."""
    extra_body = vllm_extra_body(model)
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
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
    parser = argparse.ArgumentParser(
        description="Text-output with feedback loop: ablation on PPNL single-goal test sets."
    )
    parser.add_argument("test_json",    help="Input JSON file (list of samples).")
    parser.add_argument("output_jsonl", help="Output JSONL file.")
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit number of samples processed (default: all).",
    )
    parser.add_argument(
        "--max-tries", type=int, default=3,
        help="Max inference rounds per sample with error feedback (default: 3).",
    )
    parser.add_argument(
        "--provider", choices=["openai", "vllm", "stanford"], default="openai",
        help="Model provider (default: openai).",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
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

    model = args.model

    template = load_prompt_template()
    data = json.load(open(args.test_json))
    if args.max_samples is not None:
        data = data[:args.max_samples]

    print(f"Running text-feedback (provider={args.provider}, model={model}, "
          f"max_tries={args.max_tries}) on {len(data)} samples.")

    config = {
        "_config":             True,
        "provider":            args.provider,
        "model":               model,
        "tensor_parallel_size": args.tensor_parallel_size if args.provider == "vllm" else None,
        "temperature":         TEMPERATURE,
        "max_tokens":          MAX_TOKENS,
        "max_tries":           args.max_tries,
        "prompt_file":         str(PROMPT_FILE),
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

        initial_prompt = build_initial_prompt(template, sample['nl_description'])
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
            "id":             i,
            "nl_description": sample['nl_description'],
            "ground_truth":   sample['agent_as_a_point'],
            "world":          sample['world'],
            "attempts":       attempts,
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

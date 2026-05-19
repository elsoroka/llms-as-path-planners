"""
5-shot ICL baseline inference for the PPNL single-goal benchmark.

Replicates the original gpt4.py method from the paper:
  - Model:       gpt-4
  - Temperature: 0.0
  - Max tokens:  1024
  - Prompt:      prompts/few-shot-prompts-single_goal_5.txt  (5 examples)

Usage:
    python run_baseline.py <test_json> <output_jsonl>
                           [--max-samples N]

Output JSONL: config line first, then one record per sample:
    { "id": int, "nl_description": str, "ground_truth": str, "world": list,
      "prompt": str, "response": str, "error": str|null }
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

PROMPT_FILE = Path(__file__).parent / 'prompts' / 'few-shot-prompts-single_goal_5.txt'
DEFAULT_MODEL = 'gpt-4'
TEMPERATURE   = 0.0
MAX_TOKENS    = 1024


def load_prompt_template():
    return PROMPT_FILE.read_text().rstrip()


def build_prompt(template: str, nl_description: str) -> str:
    return f"""{template}
###
Task: {nl_description}
Actions: """


def check_path(world, response: str):
    """
    Simulate the action tokens in response on the world grid.
    Ignores non-direction tokens (matches executor-point-sg.py behaviour).
    Returns None on success or a plain-English error string.
    """
    start = goal = None
    for r, row in enumerate(world):
        for c, val in enumerate(row):
            if val == 2:
                start = (r, c)
            elif val == 3:
                goal = (r, c)

    actions = [t for t in response.split() if t in ('up', 'down', 'left', 'right')]
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


def call_api(client: OpenAI, model: str, prompt: str) -> str:
    extra_body = vllm_extra_body(model)
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
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


def count_completed(output_jsonl: str) -> int:
    """Return the number of sample records already written to output_jsonl."""
    path = Path(output_jsonl)
    if not path.exists():
        return 0
    count = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not record.get("_config"):
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(
        description="5-shot ICL baseline inference on PPNL single-goal test sets."
    )
    parser.add_argument("test_json",    help="Input JSON file (list of samples).")
    parser.add_argument("output_jsonl", help="Output JSONL file.")
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit number of samples processed (default: all).",
    )
    parser.add_argument(
        "--provider", choices=["openai", "vllm", "stanford"], default="openai",
        help="Model provider (default: openai).",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
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

    model = args.model

    template = load_prompt_template()
    data     = json.load(open(args.test_json))
    if args.max_samples is not None:
        data = data[:args.max_samples]

    already_done = count_completed(args.output_jsonl)
    if already_done:
        print(f"Resuming: skipping first {already_done} samples already in output file.")

    print(f"Running 5-shot baseline (provider={args.provider}, model={model}) "
          f"on {len(data)} samples.")

    config = {
        "_config":             True,
        "provider":            args.provider,
        "model":               model,
        "tensor_parallel_size": args.tensor_parallel_size if args.provider == "vllm" else None,
        "temperature": TEMPERATURE,
        "max_tokens":  MAX_TOKENS,
        "prompt_file": str(PROMPT_FILE),
        "argv":        vars(args),
    }

    output_path = Path(args.output_jsonl)
    # Write config line only if the file doesn't already exist.
    if not output_path.exists():
        with open(output_path, 'w') as f:
            f.write(json.dumps(config) + '\n')

    new_count = 0
    for i, sample in enumerate(data):
        if i < already_done:
            continue

        print(f"\n--- Sample {i} ---")
        prompt = build_prompt(template, sample['nl_description'])
        print(f"  Prompt:\n{prompt}")

        raw = call_api(client, model, prompt)
        print(f"  Response: {raw!r}")

        error = check_path(sample['world'], raw)
        print(f"  {'Success' if error is None else 'Error: ' + error}")

        record = {
            "id":             i,
            "nl_description": sample['nl_description'],
            "ground_truth":   sample['agent_as_a_point'],
            "world":          sample['world'],
            "prompt":         prompt,
            "response":       raw,
            "error":          error,
        }

        with open(output_path, 'a') as f:
            f.write(json.dumps(record) + '\n')

        new_count += 1

    print(f"\nDone. Wrote {new_count} new samples to {args.output_jsonl} "
          f"({already_done} already existed).")


if __name__ == '__main__':
    main()

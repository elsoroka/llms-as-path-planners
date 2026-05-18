"""
5-shot ICL baseline inference for the PPNL single-goal benchmark.

Replicates the original gpt4.py method from the paper:
  - Model:       gpt-4
  - Temperature: 0.0
  - Max tokens:  250
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

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

PROMPT_FILE = Path(__file__).parent / 'prompts' / 'few-shot-prompts-single_goal_5.txt'
MODEL       = 'gpt-4'
TEMPERATURE = 0.0
MAX_TOKENS  = 250


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


def call_api(prompt: str) -> str:
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == 0:
                print(f"API error: {e}. Retrying in 25s...")
                time.sleep(25)
            else:
                raise


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
    args = parser.parse_args()

    template = load_prompt_template()
    data     = json.load(open(args.test_json))
    if args.max_samples is not None:
        data = data[:args.max_samples]

    print(f"Running 5-shot baseline (model={MODEL}) on {len(data)} samples.")

    config = {
        "_config":     True,
        "model":       MODEL,
        "temperature": TEMPERATURE,
        "max_tokens":  MAX_TOKENS,
        "prompt_file": str(PROMPT_FILE),
        "argv":        vars(args),
    }

    results = []
    for i, sample in enumerate(data):
        print(f"\n--- Sample {i} ---")
        prompt = build_prompt(template, sample['nl_description'])
        print(f"  Prompt:\n{prompt}")

        raw = call_api(prompt)
        print(f"  Response: {raw!r}")

        error = check_path(sample['world'], raw)
        print(f"  {'Success' if error is None else 'Error: ' + error}")

        results.append({
            "id":             i,
            "nl_description": sample['nl_description'],
            "ground_truth":   sample['agent_as_a_point'],
            "world":          sample['world'],
            "prompt":         prompt,
            "response":       raw,
            "error":          error,
        })

        with open(args.output_jsonl, 'w') as f:
            f.write(json.dumps(config) + '\n')
            for r in results:
                f.write(json.dumps(r) + '\n')

    print(f"\nDone. Saved {len(results)} samples to {args.output_jsonl}")


if __name__ == '__main__':
    main()

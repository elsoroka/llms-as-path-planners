import argparse
import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
from parse_code_form import parse_response, check_path
from vllm_utils import launch_vllm_server, strip_thinking, vllm_extra_body

load_dotenv()

# One-shot example task and solution (same environment used in the 5-shot baseline prompt)
ONE_SHOT_TASK = (
    "You are in a 6 by 6 world. "
    "There are obstacles that you have to avoid at: (2,1). "
    "Go from (0,1) to (3,4)"
)
ONE_SHOT_CODE = """\
def solve():
    move_x(3)
    move_y(3)"""


def build_prompt(nl_description: str) -> str:
    """
    Build the code-form one-shot prompt for a single maze task.

    Coordinate convention (matching the task descriptions):
      - Positions are given as (row, col).
      - The top-left corner is (0, 0); rows increase downward, cols increase rightward.
      - move_x(d): move d steps along columns  (+d = right, -d = left)
      - move_y(d): move d steps along rows     (+d = down = increasing row,
                                                -d = up   = decreasing row)
    """
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
        f"Task: {ONE_SHOT_TASK}\n"
        "```python\n"
        f"{ONE_SHOT_CODE}\n"
        "```\n"
        "\n"
        "Now write a function to solve this maze:\n"
        f"Task: {nl_description}\n"
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
        description="Code-form one-shot inference for PPNL single-goal benchmark."
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

    results = []
    for i, sample in enumerate(data):
        print(f"\n--- Sample {i} ---")
        initial_prompt = build_prompt(sample['nl_description'])

        # Build the message history for this sample; grows on retries.
        messages = [{"role": "user", "content": initial_prompt}]
        attempts = []

        for try_num in range(1, max_tries + 1):
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

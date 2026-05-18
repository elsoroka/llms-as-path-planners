import os
import json
import time
import sys
from openai import OpenAI
from dotenv import load_dotenv
from parse_code_form import parse_response, check_path

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# One-shot example task and solution (same environment used in the 5-shot baseline prompt)
ONE_SHOT_TASK = (
    "You are in a 6 by 6 world. "
    "There are obstacles that you have to avoid at: (2,1). "
    "Go from (0,1) to (3,4)"
)
ONE_SHOT_CODE = """\
def solve():
    # started at row 0, col 1; goal is row 3, col 4
    move_x(3)   # right 3 cols: col 1 -> col 4  (now at row 0, col 4)
    move_y(3)   # down  3 rows: row 0 -> row 3  (now at row 3, col 4 = goal)"""


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


def call_api(messages: list) -> str:
    """Call the OpenAI API with one retry on failure."""
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.0,
                max_tokens=300,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == 0:
                print(f"API error: {e}. Retrying in 25s...")
                time.sleep(25)
            else:
                raise


def main():
    """
    Usage:
        python code_form.py <test_json> <output_jsonl> [max_samples [max_tries]]

    max_tries controls how many inference rounds per sample (default 1).
    On each failed attempt the LLM receives its previous response plus a
    plain-English error description and is asked to correct the solution.

    Each output record (one JSON line) contains:
        { "id": int, "nl_description": str, "ground_truth": str, "world": list,
          "attempts": [{"try": int, "prompt": str, "response": str,
                        "error": str|null}, ...] }

    "prompt" for try 1 is the full initial prompt.
    "prompt" for tries 2+ is just the feedback message added in that turn
    (the full conversation is reconstructable from all attempts).
    """
    if len(sys.argv) < 3:
        print("Usage: python code_form.py <test_json> <output_jsonl> "
              "[max_samples [max_tries]]")
        sys.exit(1)

    test_file   = sys.argv[1]
    output_file = sys.argv[2]
    max_samples = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None
    max_tries   = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else 1

    data = json.load(open(test_file))
    if max_samples is not None:
        data = data[:max_samples]

    print(f"Running with max_tries={max_tries} on {len(data)} samples.")

    config = {
        "_config": True,
        "model":       "gpt-4",
        "temperature": 0.0,
        "argv":        sys.argv,
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

            raw = call_api(messages)
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

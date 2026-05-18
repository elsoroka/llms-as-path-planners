import os
import json
import time
import sys
from openai import OpenAI
from dotenv import load_dotenv

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


def main():
    """
    Usage:
        python code_form.py <test_json> <output_jsonl> [max_samples]

    Outputs one JSON object per line:
        { "id": int, "prompt": str, "response": str,
          "ground_truth": str, "nl_description": str, "world": list }
    """
    if len(sys.argv) < 3:
        print("Usage: python code_form.py <test_json> <output_jsonl> [max_samples]")
        sys.exit(1)

    test_file = sys.argv[1]
    output_file = sys.argv[2]
    max_samples = int(sys.argv[3]) if len(sys.argv) > 3 else None

    data = json.load(open(test_file))
    if max_samples is not None:
        data = data[:max_samples]

    results = []
    for i, sample in enumerate(data):
        print(f"\n--- Sample {i} ---")
        prompt = build_prompt(sample['nl_description'])
        print(prompt)

        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=300
                )
                break
            except Exception as e:
                if attempt == 0:
                    print(f"API error: {e}. Retrying in 25s...")
                    time.sleep(25)
                else:
                    raise

        raw = response.choices[0].message.content
        print("Response:", raw)

        result = {
            "id": i,
            "prompt": prompt,
            "response": raw,
            "ground_truth": sample['agent_as_a_point'],
            "nl_description": sample['nl_description'],
            "world": sample['world'],
        }
        results.append(result)

        # Write after each sample so progress is not lost on interruption
        with open(output_file, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')

    print(f"\nDone. Saved {len(results)} samples to {output_file}")


if __name__ == '__main__':
    main()

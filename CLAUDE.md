# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

This is a research project evaluating LLMs as path planners, based on two papers:
- *"Can Large Language Models be Good Path Planners? A Benchmark and Investigation on Spatial-temporal Reasoning"* (ICLR 2024 LLM Agents Workshop)
- *"Look Further Ahead: Testing the Limits of GPT-4 in Path Planning"* (IEEE 20th CASE 2024)

The primary active goal is to **extend the codebase with a new code-form planning prompting strategy** (see README.md for the exact specification), after first verifying the original code runs correctly.

## Repository Structure

The main code lives in the `llms-as-path-planners/` git submodule:

```
llms-as-path-planners/
├── gpt-4-path-planning/          # Long-horizon single-goal experiments
│   └── src/
│       ├── inference.py          # Main entry point; calls Azure OpenAI API
│       ├── evaluate.py           # Success rate, optimality metrics
│       ├── geometries.py         # Environments: rectangles, mazes, zig-zags
│       ├── representations.py    # Input formats: Naive, Grid, Code
│       ├── prompting.py          # Few-shot example generation
│       └── generate_samples.py   # A* pathfinding and solution generation
└── ppnl-spatial-temporal-reasoning/   # PPNL benchmark
    ├── data-synthesis/           # Dataset generation pipeline
    │   ├── generate_envs.py
    │   ├── place_agent_goals.py
    │   ├── generate_samples.py   # A* + Gurobi optimization
    │   ├── generate_all_sg_data.sh
    │   └── generate_all_mg_data.sh
    ├── ICL/                      # In-context learning inference
    │   ├── execute.py            # Main ICL runner
    │   ├── gpt4.py               # GPT-4 integration
    │   ├── hierarchical.py       # Hierarchical planning strategy
    │   ├── react.py              # ReAct prompting strategy
    │   └── prompts/              # Prompt templates
    ├── evaluate/                 # Evaluation scripts
    └── train/                    # T5/BART fine-tuning (SLURM)
```

## Dependencies and Setup

Install per-subproject; there is no top-level requirements file.

```bash
# GPT-4 path planning
pip install -r llms-as-path-planners/gpt-4-path-planning/requirements.txt
# numpy>=1.21.0, openai>=1.0.0, json5>=0.9.0

# PPNL benchmark
pip install -r llms-as-path-planners/ppnl-spatial-temporal-reasoning/requirements.txt
# Adds: gurobipy>=9.5.0, transformers, torch, datasets, accelerate, wandb
```

A `.env` file at the repo root holds the OpenAI API key. Use the **same model versions** cited in each paper.

## Running Inference

**GPT-4 path planning** (Azure OpenAI):
```bash
cd llms-as-path-planners/gpt-4-path-planning
python src/inference.py
```

**PPNL ICL experiments:**
```bash
cd llms-as-path-planners/ppnl-spatial-temporal-reasoning/ICL
python execute.py
```

**PPNL data synthesis:**
```bash
cd llms-as-path-planners/ppnl-spatial-temporal-reasoning/data-synthesis
bash generate_all_sg_data.sh   # single-goal
bash generate_all_mg_data.sh   # multi-goal
```

## Data Format

Grid worlds encode cells as integers. Natural language + solution paths are stored together:
```json
{
  "world": [[0,0,0,2,0], [0,3,0,0,1]],
  "nl_description": "You are in a 6 by 6 world. There are obstacles at: (5,3). Go from (1,4) to (2,1)",
  "solution_coordinates": [[1,4],[1,3],[1,2],[1,1],[2,1]],
  "agent_as_a_point": "left left left down",
  "agent_has_direction": "turn right move forward ..."
}
```

Output from new experiments should be saved as `.jsonl` (one sample per line) including: unique id, exact prompt, and raw LLM text response.

## Implemented Strategies

### Code-Form Planning (DONE)

Both subprojects have a fully-implemented code-form strategy that asks the LLM to write a Python `solve()` function using `move_x(dist)` / `move_y(dist)` instead of step-by-step direction tokens.

**Files:**
- `gpt-4-path-planning/src/code_form.py` — inference; supports `--representation {text,code,grid}`, `--max-tries N`
- `gpt-4-path-planning/src/parse_code_form.py` — parser + `check_path` validator; outputs evaluated JSONL
- `gpt-4-path-planning/src/run_code_form.sh` — loops over all 6 geometry×split combos
- `ppnl-spatial-temporal-reasoning/ICL/code_form.py` — inference; single-representation (NL description)
- `ppnl-spatial-temporal-reasoning/ICL/parse_code_form.py` — parser + `check_path` validator
- `ppnl-spatial-temporal-reasoning/ICL/run_code_form.sh` — loops over 5 PPNL test sets

**Prompt structure (one-shot):** Fixed example (25×25 rectangle obstacle, start (2,5), goal (12,5)) demonstrating the `solve()` function. For PPNL, the example uses a 6×6 task.

**Retry/feedback loop:**
- On each failed attempt, the error from `check_path` is sent back as a user message:
  ```
  That solution is incorrect.
  Error: <error>
  Please write a corrected solve() function.
  ```python
  ```
- Multi-turn conversation history grows with each retry.
- `--max-tries` (default 3 for PPNL, 7 for gpt-4) controls max rounds.

**Output JSONL format:**
```json
{"_config": true, "model": "...", "temperature": 0.0, ...}
{"id": 0, "naive_representation": "...", "ground_truth": "...", "world": [...],
 "attempts": [{"try": 1, "prompt": "...", "response": "...", "error": "..."},
              {"try": 2, "prompt": "...", "response": "...", "error": null}]}
```

**`check_path` error messages** (identical in both subprojects):
- `"No valid move_x/move_y calls parsed from response"`
- `"Out of bounds at step N: tried to move to (r, c)"`
- `"Hit obstacle at step N: (r, c)"`
- `"Path ended at (r, c) but goal is at (gr, gc)"`

### Text-Output Baseline (DONE)

**PPNL:** `ppnl-spatial-temporal-reasoning/ICL/run_baseline.py` — 5-shot ICL, single call, no feedback. Prompt loads from `prompts/few-shot-prompts-single_goal_5.txt`; appends `Task: {nl}\nActions: `.

**gpt-4:** `gpt-4-path-planning/src/inference.py` — n-shot prompting via `prompting.py` with random exemplar selection from training split. Representation: Naive (text description → direction tokens).

### Text-Output with Feedback Ablation (DONE)

Goal: same feedback loop as code-form, but for the standard text output (direction tokens). Lets us isolate the effect of the code representation vs. the feedback loop alone.

**Design:**
- Uses the **same prompt as the text baseline** (5-shot for PPNL, one-shot for gpt-4).
- Parses the response by extracting direction tokens (`up`/`down`/`left`/`right`).
- Validates with `check_path` — produces **identical error messages** as the code-form variant.
- Feedback message:
  ```
  That solution is incorrect.
  Error: <error>
  Please write a corrected action sequence.
  Actions: 
  ```
- Same retry/conversation-history structure as `code_form.py`.
- Same JSONL output format (with `"attempts"` list).

**Files:**
- `ppnl-spatial-temporal-reasoning/ICL/text_feedback.py` — inference; 5-shot template + retry loop
- `ppnl-spatial-temporal-reasoning/ICL/parse_text_feedback.py` — parser + check_path for text output
- `ppnl-spatial-temporal-reasoning/ICL/run_text_feedback.sh` — loops over 5 PPNL test sets
- `gpt-4-path-planning/src/text_feedback.py` — inference; fixed one-shot + retry loop; `--representation {text,code,grid}`
- `gpt-4-path-planning/src/parse_text_feedback.py` — parser + check_path for text output
- `gpt-4-path-planning/src/run_text_feedback.sh` — loops over all 6 geometry×split combos

**gpt-4 one-shot example** (same scenario as code_form.py):
- 25×25, obstacle block rows 5–9 cols 3–7, start (2,5), goal (12,5)
- Solution: `down down right right right right right down down down down down down down down left left left left left`

## Key Architecture Notes

- **Pathfinding**: A* with Manhattan distance heuristic throughout; multi-goal cases use Gurobi for optimal ordering.
- **LLM calls**: Temperature 0.0 (deterministic), max tokens 200, frequency penalty 0.25.
- **Test splits**: IID (path lengths 2–25), OOD (25–200); seen/unseen obstacle configurations.
- **Grid sizes evaluated**: 5×5, 6×6, 7×7, and 6×6 with more obstacles.
- **World encoding**: 0=free, 1=obstacle, 2=start, 3=goal.
- **run_baseline.py / run_code_form.sh**: hardcode all parameters; loop over test-set lists. Follow this pattern for new scripts.

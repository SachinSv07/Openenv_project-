# OpenEnv Customer Support Ticket Resolution Environment

A complete OpenEnv-compatible environment for evaluating AI agents on customer support workflows.

## Problem Description

This environment simulates a realistic customer support queue where an agent must:

1. classify each ticket
2. choose an appropriate response
3. escalate when priority requires it

The environment is step-based and deterministic, making it suitable for repeatable benchmarking and CI evaluation.

## Why This Matters

Customer support automation is a practical real-world AI use case. Strong support agents need to balance:

- intent understanding (classification)
- communication quality (response selection)
- risk handling (escalation)

This benchmark captures all three.

## Action Space

`Action` fields:

- `action_type`: one of `classify`, `respond`, `escalate`
- `category`: optional predicted category in `{delivery, refund, technical, general}`
- `response`: optional response text

## Observation Structure

`Observation` fields:

- `current_ticket`: current `PublicTicket`
  - `id`: int
  - `content`: str
  - `priority`: str in `{low, medium, high}`
- `history`: list of prior `Action` objects

Ground-truth `category` is hidden from the observation and used internally by the grader.

## Tasks

### Easy
- objective: classify ticket correctly
- reward:
  - correct classification: `1.0`
  - wrong/missing classification: `0.0`

### Medium
- objective: classify + provide suitable response
- reward:
  - classification: `0.5`
  - response quality: `0.5` (keyword-based deterministic grading)

### Hard
- objective: classify + respond + escalate correctly for high-priority tickets
- reward:
  - classification: `0.3`
  - response quality: `0.4`
  - escalation correctness: `0.3`

## Deterministic Grading

- category: exact match
- response quality: keyword matching against category-specific vocab
- escalation:
  - high priority requires `action_type == "escalate"`
  - non-high priority should not escalate

All per-step scores are bounded to `[0.0, 1.0]`.

## Project Structure

```
openenv_support_env/
├── env.py
├── models.py
├── tasks/
│   ├── easy.py
│   ├── medium.py
│   └── hard.py
├── graders/
│   ├── easy_grader.py
│   ├── medium_grader.py
│   └── hard_grader.py
├── data/
│   └── tickets.json
├── baseline.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Baseline

From `openenv_support_env`:

```bash
python baseline.py --task hard
```

You can also evaluate other tasks:

```bash
python baseline.py --task easy
python baseline.py --task medium
```

## Example Output (Hard Task)

```
Ticket  1 | task=hard   | score=1.00 | reason=Classification, response, and escalation are all correct.
Ticket  2 | task=hard   | score=1.00 | reason=Classification, response, and escalation are all correct.
...
--------------------------------------------------------------------------------
Processed tickets: 18
Total score: 17.40
Average score: 0.967
```

(Exact scores depend on the baseline heuristic decisions, but are fully reproducible.)

## Docker

Build and run:

```bash
docker build -t openenv-support-env .
docker run --rm openenv-support-env
```

Default container command runs:

```bash
python baseline.py --task hard
```

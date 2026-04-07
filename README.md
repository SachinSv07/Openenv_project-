---
title: Customer Support OpenEnv
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_file: baseline.py
pinned: false
---
# Customer Support AI Evaluation Environment (OpenEnv)

## Overview

This repository contains a realistic OpenEnv-compatible benchmark for customer support automation.

Agents are evaluated on their ability to:

1. classify support tickets,
2. generate useful responses,
3. decide when escalation is required.

The benchmark is designed to assess reasoning and decision quality, not shallow keyword matching.

## Motivation

Customer support is a high-impact AI use case where errors directly affect trust, retention, and operational cost.

Most benchmarks focus on isolated classification accuracy. This environment evaluates full triage behavior: categorization, response quality, escalation decisions, history-aware consistency, and efficiency.

## Why This Environment Matters

- Customer support is a real-world, high-impact domain where AI systems must combine reasoning, language understanding, and decision-making.
- This environment evaluates not just correctness, but quality of responses, escalation decisions, and efficiency.
- Unlike simple benchmarks, it discourages shortcut strategies such as keyword matching or reward hacking.
- It provides a more realistic measure of how AI agents perform in production-like workflows.

## Environment Design

### Observation Space

- `current_ticket` with public fields only:
  - `id`
  - `content`
  - `priority`
- `history` of prior actions

Ground-truth category is hidden from observations to prevent leakage.

### Action Space

- `action_type`: `classify` | `respond` | `escalate`
- `category`: `delivery` | `refund` | `technical` | `general`
- `response`: free-text response
- `escalate`: optional boolean escalation signal

### Reward System

Hard-task composition:

- classification: up to `0.3`
- response quality: up to `0.4`
- escalation decision: up to `0.3`

Additional shaping:

- penalties for mistakes (wrong class, contradiction, escalation errors)
- history-aware penalties for repeated mistakes
- efficiency shaping with a strict one-step bonus

All final scores are clipped to `[0.0, 1.0]`.

## Tasks

### Easy

Classification only.

### Medium

Classification + response quality.

### Hard

Classification + response + escalation + efficiency + history-aware penalties.

## Dataset

The dataset contains 20+ realistic tickets featuring:

- ambiguous phrasing,
- mixed intents,
- emotional tone,
- practical production-like support scenarios.

Representative cases include mixed delivery/billing concerns, confirmation follow-ups, repeated frustration, and intentionally ambiguous requests.

## Example Output

Sample hard-task output (abridged):

```text
Ticket  2 | Predicted: refund | Escalated: no | Score: 0.60
Reason: Classification is correct. Response is generic and lacks actionable detail (0.1). ...

Ticket  5 | Predicted: technical | Escalated: yes | Score: 0.15
Reason: Classification is not exact, but partially plausible ... Wrong classification penalty applied ...

Ticket  9 | Predicted: delivery | Escalated: yes | Score: 1.00
Reason: Classification is correct. Strong response with issue-specific handling and actionable detail (0.4). ...

Processed tickets: 24
Average score: ~0.63
```

## Setup Instructions

```bash
pip install -r requirements.txt
python baseline.py --task hard
```

## Inference

The required inference entry point is `inference.py` at repository root.

Required environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

## Docker Instructions

```bash
docker build -t support-env .
docker run support-env
```

## Hugging Face Deployment

1. Push this repository to Hugging Face Spaces.
2. Create a Space using the Docker SDK.
3. Keep `Dockerfile` at repository root.
4. Add topic/tag metadata including `openenv`.

## Baseline Results

Current baseline average is typically around `0.6` to `0.7` on the hard task.

The baseline agent intentionally uses simple heuristic rules and is not optimized for the grading system. This ensures the environment meaningfully differentiates stronger reasoning-based agents from weaker ones, preventing inflated benchmark scores.

The baseline is intentionally imperfect because:

- heuristics are simple,
- responses are often generic,
- ambiguous tickets trigger partial-credit and penalty cases.

## Key Features

- no ground-truth leakage in observation
- nuanced reward shaping
- explainable grading reasons
- history-aware penalty logic
- efficiency-based scoring
- deterministic evaluation behavior
- Designed to prevent reward hacking and keyword-based exploitation through hidden ground-truth labels and strict, category-aware grading.

## Limitations

- Rule-based grading may not fully capture nuanced human judgment.
- Dataset size is limited but designed for diversity and ambiguity.
- Future work can include LLM-based evaluation or larger datasets.

## Project Structure

```text
openenvhackathon/
├── baseline.py
├── data/
├── Dockerfile
├── env.py
├── graders/
├── inference.py
├── models.py
├── openenv.yaml
├── README.md
├── requirements.txt
├── scripts/
└── tasks/
```

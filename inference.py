from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

from openai import OpenAI

from env import SupportEnv
from models import Action, PublicTicket

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("API_KEY", ""))
TASK_NAME = os.getenv("OPENENV_TASK", "hard")
BENCHMARK = os.getenv("OPENENV_BENCHMARK", "customer-support-env")
MAX_STEPS = int(os.getenv("MAX_STEPS", "50"))


def _print_start() -> None:
    sys.stdout.write(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}\n")
    sys.stdout.flush()


def _print_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_value = "null" if error is None else error.replace("\n", " ").replace("\r", " ")
    done_value = "true" if done else "false"
    sys.stdout.write(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_value} error={error_value}\n"
    )
    sys.stdout.flush()


def _print_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    success_value = "true" if success else "false"
    rewards_value = ",".join(f"{reward:.2f}" for reward in rewards)
    sys.stdout.write(
        f"[END] success={success_value} steps={steps} score={score:.2f} rewards={rewards_value}\n"
    )
    sys.stdout.flush()


def _build_client() -> Optional[OpenAI]:
    if not HF_TOKEN:
        return None
    return OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


def _short_text(text: str, max_len: int = 120) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def _local_policy(ticket: PublicTicket) -> Action:
    text = ticket.content.lower()
    if any(k in text for k in ["money", "refund", "charged", "billed", "return", "damaged"]):
        category = "refund"
        response = "I understand the issue and can review your refund request."
    elif any(k in text for k in ["error", "crash", "app", "login", "payment", "update", "otp", "freeze", "patch"]):
        category = "technical"
        response = "Please try restarting the app and check whether the issue continues."
    elif any(k in text for k in ["order", "arriv", "delivery", "shipping", "package", "courier", "tracking"]):
        category = "delivery"
        response = "Sorry for the delay. We will check the shipping status and follow up."
    else:
        category = "general"
        response = "Thanks for reaching out. We will look into this request."

    escalate = ticket.priority == "high"
    return Action(action_type="escalate" if escalate else "respond", category=category, response=response, escalate=escalate)


def _llm_policy(client: OpenAI, ticket: PublicTicket) -> Action:
    prompt = {
        "ticket": {
            "id": ticket.id,
            "content": ticket.content,
            "priority": ticket.priority,
        },
        "instructions": (
            "Return JSON with keys action_type, category, response, escalate. "
            "Choose the best category from delivery/refund/technical/general. "
            "Use escalate=true for high priority tickets."
        ),
    }

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.0,
            max_tokens=220,
            messages=[
                {"role": "system", "content": "You are a precise customer support triage assistant."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
        )
        content = completion.choices[0].message.content or ""
        payload = json.loads(content)
        action_type = str(payload.get("action_type", "respond"))
        category = payload.get("category")
        response = payload.get("response")
        escalate = payload.get("escalate")
        if isinstance(escalate, str):
            escalate = escalate.lower() == "true"
        elif not isinstance(escalate, bool):
            escalate = ticket.priority == "high"

        if action_type not in {"classify", "respond", "escalate"}:
            action_type = "respond"
        if category not in {"delivery", "refund", "technical", "general"}:
            category = "general"
        if not isinstance(response, str) or not response.strip():
            response = _local_policy(ticket).response

        return Action(action_type=action_type, category=category, response=response, escalate=escalate)
    except Exception:
        return _local_policy(ticket)


def run_inference(task_level: str = TASK_NAME) -> None:
    env = SupportEnv(task_level=task_level)
    client = _build_client()
    observation = env.reset()
    rewards: List[float] = []
    steps = 0
    success = False
    error: Optional[str] = None

    _print_start()

    try:
        done = False
        while not done and steps < MAX_STEPS:
            ticket = observation.current_ticket
            action = _llm_policy(client, ticket) if client is not None else _local_policy(ticket)
            action_str = json.dumps(action.model_dump(), ensure_ascii=False, separators=(",", ":"))
            observation, reward, done, _info = env.step(action)
            rewards.append(reward.score)
            steps += 1
            _print_step(steps, action_str, reward.score, done, None)

        success = done
    except Exception as exc:
        error = str(exc)
        _print_step(steps + 1, "{}", 0.00, False, error)
    finally:
        env.close()
        score = sum(rewards) / len(rewards) if rewards else 0.0
        _print_end(success and error is None, steps, score, rewards)


if __name__ == "__main__":
    run_inference()

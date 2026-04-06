from __future__ import annotations

import argparse
from typing import Dict

from env import SupportEnv
from models import Action, PublicTicket


def classify_ticket(content: str) -> str:
    text = content.lower()

    if any(k in text for k in ["money", "refund", "charged", "billed", "return", "damaged"]):
        return "refund"
    if any(k in text for k in ["error", "crash", "app", "login", "payment", "update", "otp", "freez", "patch"]):
        return "technical"
    if any(k in text for k in ["order", "arriv", "delivery", "shipping", "package", "courier", "tracking"]):
        return "delivery"
    return "general"


def generate_response(ticket: PublicTicket, category: str) -> str:
    templates: Dict[str, str] = {
        "delivery": "Sorry for the delay. We will check your shipping status and follow up.",
        "refund": "I understand the billing concern and we can review this request.",
        "technical": "Please try restarting once and check if the issue continues.",
        "general": "We have noted your request and will respond soon.",
    }
    return templates[category]


def choose_action(ticket: PublicTicket) -> Action:
    category = classify_ticket(ticket.content)
    response = generate_response(ticket, category)
    action_type = "escalate" if ticket.priority == "high" else "respond"

    return Action(action_type=action_type, category=category, response=response)


def shorten(text: str, max_len: int = 72) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3] + "..."


def run_baseline(task_level: str) -> None:
    env = SupportEnv(task_level=task_level)
    observation = env.reset()

    total_score = 0.0
    steps = 0

    done = False
    while not done:
        current_ticket = observation.current_ticket
        action = choose_action(current_ticket)
        observation, reward, done, info = env.step(action)
        total_score += reward.score
        steps += 1

        escalated = "yes" if action.action_type == "escalate" else "no"
        print(
            f"Ticket {info['ticket_id']:>2} | Content: \"{shorten(current_ticket.content)}\"\n"
            f"  Predicted: {action.category} | Escalated: {escalated} | "
            f"Score: {reward.score:.2f} | Base: {info['base_score']:.2f}\n"
            f"  Reason: {reward.reason}\n"
        )

    avg_score = total_score / max(steps, 1)
    print("-" * 80)
    print(f"Processed tickets: {steps}")
    print(f"Total score: {total_score:.2f}")
    print(f"Average score: {avg_score:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline agent for OpenEnv Support Environment")
    parser.add_argument(
        "--task",
        type=str,
        default="hard",
        choices=["easy", "medium", "hard"],
        help="Task difficulty to evaluate",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_baseline(task_level=args.task)

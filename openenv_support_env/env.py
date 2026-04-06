from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from graders.easy_grader import grade_easy_detailed
from graders.hard_grader import grade_hard_detailed
from graders.medium_grader import grade_medium_detailed
from graders.common import GraderResult, clip_score
from models import Action, Observation, PublicTicket, Reward, Ticket


class SupportEnv:
    def __init__(self, task_level: str = "easy", data_path: Optional[str] = None):
        self.task_level = task_level.lower()
        self.data_path = Path(data_path) if data_path else Path(__file__).parent / "data" / "tickets.json"

        self._task_graders: Dict[str, Callable[[Ticket, Action], GraderResult]] = {
            "easy": grade_easy_detailed,
            "medium": grade_medium_detailed,
            "hard": grade_hard_detailed,
        }

        if self.task_level not in self._task_graders:
            raise ValueError("task_level must be one of: easy, medium, hard")

        self.tickets: List[Ticket] = []
        self.current_index: int = 0
        self.history: List[Action] = []
        self.actions_on_current_ticket: int = 0
        self.wrong_classification_streak: int = 0
        self.unnecessary_escalation_streak: int = 0
        self.unnecessary_escalations_total: int = 0

    @staticmethod
    def _to_public_ticket(ticket: Ticket) -> PublicTicket:
        return PublicTicket(id=ticket.id, content=ticket.content, priority=ticket.priority)

    def _load_tickets(self) -> List[Ticket]:
        with self.data_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        tickets = [Ticket(**item) for item in raw]
        tickets.sort(key=lambda t: t.id)
        return tickets

    def reset(self) -> Observation:
        self.tickets = self._load_tickets()
        self.current_index = 0
        self.history = []
        self.actions_on_current_ticket = 0
        self.wrong_classification_streak = 0
        self.unnecessary_escalation_streak = 0
        self.unnecessary_escalations_total = 0

        if not self.tickets:
            raise ValueError("No tickets found in dataset.")

        return self.state()

    def state(self) -> Observation:
        if self.current_index >= len(self.tickets):
            raise IndexError("Environment episode has ended. Call reset() to start a new episode.")

        return Observation(
            current_ticket=self._to_public_ticket(self.tickets[self.current_index]),
            history=list(self.history),
        )

    def step(self, action: Action) -> Tuple[Optional[Observation], Reward, bool, dict]:
        if self.current_index >= len(self.tickets):
            raise IndexError("Cannot call step() after episode is done. Call reset() first.")

        ticket = self.tickets[self.current_index]
        grader = self._task_graders[self.task_level]
        self.actions_on_current_ticket += 1

        grader_result = grader(ticket, action)
        base_score = grader_result["score"]
        base_reason = grader_result["reason"]
        flags = grader_result["flags"]

        history_penalty, history_reason = self._history_penalty(flags)
        resolved = self._is_ticket_resolved(action)
        response_component = float(grader_result["breakdown"].get("response", 0.0))
        efficiency_adjustment, efficiency_reason = self._efficiency_adjustment(resolved, base_score, response_component)

        final_score = clip_score(base_score + history_penalty + efficiency_adjustment)
        reason = self._compose_reason(base_reason, history_reason, efficiency_reason)
        reward = Reward(score=final_score, reason=reason)

        self.history.append(action)

        actions_used = self.actions_on_current_ticket
        should_advance = resolved or self.actions_on_current_ticket >= 3
        if should_advance:
            self.current_index += 1
            self.actions_on_current_ticket = 0

        done = self.current_index >= len(self.tickets)
        next_obs = None if done else self.state()
        info = {
            "ticket_id": ticket.id,
            "task_level": self.task_level,
            "progress": f"{self.current_index}/{len(self.tickets)}",
            "resolved": should_advance,
            "actions_on_ticket": actions_used,
            "base_score": base_score,
            "history_penalty": history_penalty,
            "efficiency_adjustment": efficiency_adjustment,
            "breakdown": grader_result["breakdown"],
        }

        return next_obs, reward, done, info

    def _is_ticket_resolved(self, action: Action) -> bool:
        if self.task_level == "easy":
            return action.category is not None
        return (action.category is not None) and bool(action.response)

    def _history_penalty(self, flags: Dict[str, bool]) -> tuple[float, str]:
        penalty = 0.0
        reasons: List[str] = []

        if flags.get("wrong_classification", False):
            self.wrong_classification_streak += 1
            if self.wrong_classification_streak >= 2:
                streak_penalty = -0.05 * (self.wrong_classification_streak - 1)
                penalty += streak_penalty
                reasons.append(
                    f"Repeated wrong classification streak penalty applied ({streak_penalty:.2f})."
                )
        else:
            self.wrong_classification_streak = 0

        if flags.get("unnecessary_escalation", False):
            self.unnecessary_escalation_streak += 1
            self.unnecessary_escalations_total += 1

            if self.unnecessary_escalation_streak >= 3:
                streak_penalty = -0.05 * (self.unnecessary_escalation_streak - 2)
                penalty += streak_penalty
                reasons.append(f"Repeated unnecessary escalation streak penalty applied ({streak_penalty:.2f}).")

            if self.unnecessary_escalations_total >= 3:
                cumulative_penalty = -0.03 * (self.unnecessary_escalations_total - 2)
                penalty += cumulative_penalty
                reasons.append(
                    f"Cumulative unnecessary escalation penalty applied ({cumulative_penalty:.2f})."
                )
        else:
            self.unnecessary_escalation_streak = 0

        if not reasons:
            return 0.0, "No history-based penalties applied."
        return penalty, " ".join(reasons)

    def _efficiency_adjustment(self, resolved: bool, base_score: float, response_component: float) -> tuple[float, str]:
        if (
            resolved
            and self.actions_on_current_ticket == 1
            and base_score >= 0.5
            and response_component >= 0.2
        ):
            return 0.1, "Efficiency bonus applied due to effective one-step resolution (+0.10)."
        if resolved and self.actions_on_current_ticket == 1 and base_score < 0.5:
            return 0.0, "No efficiency bonus: one-step action quality was below threshold (base_score < 0.5)."
        if resolved and self.actions_on_current_ticket == 1 and response_component < 0.2:
            return 0.0, "No efficiency bonus: response quality was too generic for a one-step reward."
        if resolved and self.actions_on_current_ticket > 1:
            penalty = -0.1 * (self.actions_on_current_ticket - 1)
            return penalty, f"Efficiency penalty applied for extra actions ({penalty:.2f})."
        if not resolved and self.actions_on_current_ticket >= 3:
            return -0.1, "Ticket not resolved within action limit; efficiency penalty applied (-0.10)."
        return -0.02, "Ticket not yet resolved this step; minor efficiency penalty applied (-0.02)."

    @staticmethod
    def _compose_reason(base_reason: str, history_reason: str, efficiency_reason: str) -> str:
        return f"{base_reason} {history_reason} {efficiency_reason}"

    def close(self) -> None:
        """No-op close method for benchmark compatibility."""
        return None

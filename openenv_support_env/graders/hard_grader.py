from models import Action, Ticket
from graders.common import (
    GraderResult,
    classification_component,
    clip_score,
    contradiction_penalty,
    response_feedback,
    response_score_hard,
)


def _escalation_analysis(ticket: Ticket, action: Action) -> tuple[float, float, bool, str]:
    is_high_priority = ticket.priority == "high"
    escalated = action.action_type == "escalate"

    if is_high_priority and escalated:
        return 0.3, 0.0, False, "Escalation was correctly used for a high-priority ticket."
    if is_high_priority and (not escalated):
        return 0.0, -0.3, False, "Required escalation was missed for a high-priority ticket (-0.3)."
    if (not is_high_priority) and escalated:
        return 0.1, -0.1, True, "Escalation was unnecessary for non-high priority (-0.1)."
    if (not is_high_priority) and (not escalated):
        return 0.3, 0.0, False, "No escalation was needed and none was used."
    return 0.0, 0.0, False, "Escalation analysis completed."


def grade_hard_detailed(ticket: Ticket, action: Action) -> GraderResult:
    classification_score, classification_reason, wrong_classification = classification_component(
        ticket, action, max_score=0.3
    )
    response_score = response_score_hard(ticket, action)
    escalation_score, escalation_penalty, unnecessary_escalation, escalation_reason = _escalation_analysis(ticket, action)
    contradiction = contradiction_penalty(ticket, action)
    wrong_penalty = -0.3 if wrong_classification else 0.0

    raw_total = (
        classification_score
        + response_score
        + escalation_score
        + escalation_penalty
        + contradiction
        + wrong_penalty
    )
    final_score = clip_score(raw_total)

    reason_parts = [
        classification_reason,
        response_feedback(ticket, action, response_score),
        escalation_reason,
    ]
    if contradiction < 0:
        reason_parts.append("Contradictory response language penalty applied (-0.2).")
    if wrong_penalty < 0:
        reason_parts.append("Wrong classification penalty applied (-0.3).")
    if contradiction == 0.0 and wrong_penalty == 0.0 and escalation_penalty == 0.0:
        reason_parts.append("No penalties applied.")

    return {
        "score": final_score,
        "reason": " ".join(reason_parts),
        "breakdown": {
            "classification": classification_score,
            "response": response_score,
            "escalation": escalation_score,
            "wrong_classification_penalty": wrong_penalty,
            "escalation_penalty": escalation_penalty,
            "contradiction_penalty": contradiction,
            "raw_total": raw_total,
            "final": final_score,
        },
        "flags": {
            "wrong_classification": wrong_classification,
            "unnecessary_escalation": unnecessary_escalation,
        },
    }


def grade_hard(ticket: Ticket, action: Action) -> float:
    """Compatibility helper returning only score."""
    return grade_hard_detailed(ticket, action)["score"]

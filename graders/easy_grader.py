from models import Action, Ticket
from graders.common import GraderResult, classification_component, clip_score


def grade_easy_detailed(ticket: Ticket, action: Action) -> GraderResult:
	classification_score, classification_reason, wrong_classification = classification_component(
		ticket, action, max_score=1.0
	)

	wrong_penalty = -0.3 if wrong_classification else 0.0
	raw_total = classification_score + wrong_penalty
	final_score = clip_score(raw_total)

	reason_parts = [classification_reason]
	if wrong_penalty < 0:
		reason_parts.append("Wrong classification penalty applied (-0.3).")
	else:
		reason_parts.append("No penalty applied.")

	return {
		"score": final_score,
		"reason": " ".join(reason_parts),
		"breakdown": {
			"classification": classification_score,
			"wrong_classification_penalty": wrong_penalty,
			"raw_total": raw_total,
			"final": final_score,
		},
		"flags": {
			"wrong_classification": wrong_classification,
			"unnecessary_escalation": False,
		},
	}


def grade_easy(ticket: Ticket, action: Action) -> float:
	"""Compatibility helper returning only score."""
	return grade_easy_detailed(ticket, action)["score"]

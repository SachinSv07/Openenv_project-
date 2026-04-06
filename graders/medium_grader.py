from models import Action, Ticket
from graders.common import (
	GraderResult,
	classification_component,
	clip_score,
	contradiction_penalty,
	response_feedback,
	response_score_medium,
)


def grade_medium_detailed(ticket: Ticket, action: Action) -> GraderResult:
	classification_score, classification_reason, wrong_classification = classification_component(
		ticket, action, max_score=0.5
	)
	response_score = response_score_medium(ticket, action)
	contradiction = contradiction_penalty(ticket, action)
	wrong_penalty = -0.3 if wrong_classification else 0.0

	raw_total = classification_score + response_score + contradiction + wrong_penalty
	final_score = clip_score(raw_total)

	reason_parts = [classification_reason, response_feedback(ticket, action, response_score)]
	if contradiction < 0:
		reason_parts.append("Contradictory response language penalty applied (-0.2).")
	if wrong_penalty < 0:
		reason_parts.append("Wrong classification penalty applied (-0.3).")
	if not contradiction and not wrong_penalty:
		reason_parts.append("No penalties applied.")

	return {
		"score": final_score,
		"reason": " ".join(reason_parts),
		"breakdown": {
			"classification": classification_score,
			"response": response_score,
			"wrong_classification_penalty": wrong_penalty,
			"contradiction_penalty": contradiction,
			"raw_total": raw_total,
			"final": final_score,
		},
		"flags": {
			"wrong_classification": wrong_classification,
			"unnecessary_escalation": False,
		},
	}


def grade_medium(ticket: Ticket, action: Action) -> float:
	"""Compatibility helper returning only score."""
	return grade_medium_detailed(ticket, action)["score"]

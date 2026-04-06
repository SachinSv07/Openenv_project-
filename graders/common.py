from __future__ import annotations

from typing import Dict, List, Set, TypedDict

from models import Action, Ticket

DELIVERY_STATUS_KEYWORDS = ["delay", "track", "shipping"]
TECHNICAL_TROUBLESHOOTING_KEYWORDS = ["try", "restart", "check", "steps"]
GENERAL_TONE_KEYWORDS = ["thank", "happy", "assist", "glad", "help"]

CATEGORY_HINTS: Dict[str, List[str]] = {
	"delivery": ["order", "delivery", "shipping", "track", "courier", "package", "arriv"],
	"refund": ["refund", "return", "money back", "reimburse", "charged", "billed", "credit"],
	"technical": ["crash", "error", "bug", "issue", "login", "freez", "otp", "update", "steps"],
	"general": ["plan", "hours", "invoice", "help", "confirm", "clarify", "membership"],
}


class GraderResult(TypedDict):
	score: float
	reason: str
	breakdown: Dict[str, float]
	flags: Dict[str, bool]


CONTRADICTION_KEYWORDS: Dict[str, List[str]] = {
	"refund": ["refund", "return", "money back", "reimburse"],
}


def clip_score(score: float) -> float:
	return max(0.0, min(1.0, score))


def _contains_any(text: str, keywords: List[str]) -> bool:
	return any(keyword in text for keyword in keywords)


def detect_signal_categories(text: str) -> Set[str]:
	lower = text.lower()
	detected: Set[str] = set()
	for category, hints in CATEGORY_HINTS.items():
		if _contains_any(lower, hints):
			detected.add(category)
	return detected


def classification_component(ticket: Ticket, action: Action, max_score: float) -> tuple[float, str, bool]:
	"""Granular classification score; returns (score, explanation, wrong_flag)."""
	predicted = action.category
	if predicted is None:
		return 0.0, "Classification missing.", True

	if predicted == ticket.category:
		return max_score, "Classification is correct.", False

	signals = detect_signal_categories(ticket.content)
	if predicted in signals and len(signals) >= 2:
		partial = round(max_score * 0.67, 2)
		return partial, "Classification is not exact, but partially plausible for mixed-intent content.", True
	if predicted in signals:
		partial = round(max_score * 0.34, 2)
		return partial, "Classification is incorrect but weakly related to ticket wording.", True
	return 0.0, "Classification is incorrect.", True


def response_score_medium(ticket: Ticket, action: Action) -> float:
	"""Tiered response scoring for medium task.

	Levels for delivery/refund/technical: strong=0.4, partial=0.2, generic=0.1, irrelevant=0.0
	General category max is 0.2.
	"""
	return _tiered_response_score(ticket, action)


def response_score_hard(ticket: Ticket, action: Action) -> float:
	"""Tiered response scoring for hard task.

	Levels for delivery/refund/technical: strong=0.4, partial=0.2, generic=0.1, irrelevant=0.0
	General category max is 0.2.
	"""
	return _tiered_response_score(ticket, action)


def _tiered_response_score(ticket: Ticket, action: Action) -> float:
	response = (action.response or "").lower().strip()
	if not response:
		return 0.0

	action_words = ["check", "update", "follow", "investigat", "steps", "process", "initiated", "timeline"]

	if ticket.category == "delivery":
		has_apology = "sorry" in response or "apolog" in response
		has_delivery_context = _contains_any(response, ["delivery", "shipping", "track", "delay", "courier", "package"])
		has_required_delivery = _contains_any(response, ["delay", "track", "shipping"])
		has_action = _contains_any(response, action_words)

		if has_apology and has_required_delivery and has_action:
			return 0.4
		if has_delivery_context and (has_apology or has_action):
			return 0.2
		if "we will check" in response or "check this" in response:
			return 0.1
		return 0.0

	if ticket.category == "refund":
		has_refund = "refund" in response
		has_process = _contains_any(response, ["process", "initiated", "timeline"])
		has_apology = "sorry" in response or "apolog" in response

		if has_refund and has_process:
			return 0.4
		if has_refund:
			return 0.2
		if has_apology and ("refund" not in response):
			return 0.1
		return 0.0

	if ticket.category == "technical":
		has_troubleshooting = _contains_any(response, ["restart", "clear cache", "try again", "check", "steps", "try"])
		has_issue_context = _contains_any(response, ["issue", "error", "bug", "technical", "problem", "fix"])

		if has_troubleshooting and has_issue_context:
			return 0.4
		if has_issue_context:
			return 0.2
		if "we will fix" in response or "we'll fix" in response:
			return 0.1
		return 0.0

	has_helpful_tone = _contains_any(response, GENERAL_TONE_KEYWORDS)
	has_specific_help = _contains_any(response, ["clarify", "guide", "check", "share details", "confirm"])
	if has_helpful_tone and has_specific_help:
		return 0.2
	if has_helpful_tone:
		return 0.1
	return 0.0


def contradiction_penalty(ticket: Ticket, action: Action) -> float:
	"""Penalty for contradictory category language in response (e.g., refund text on non-refund ticket)."""
	response = (action.response or "").lower().strip()
	if not response:
		return 0.0

	for category, keywords in CONTRADICTION_KEYWORDS.items():
		if category == ticket.category:
			continue
		if _contains_any(response, keywords):
			return -0.2
	return 0.0


def response_feedback(ticket: Ticket, action: Action, response_score: float) -> str:
	if not action.response:
		return "Response is missing."

	if ticket.category == "general":
		if response_score >= 0.2:
			return "Helpful and relevant general response (0.2)."
		if response_score >= 0.1:
			return "General response is generic but polite (0.1)."
		return "General response is irrelevant or unhelpful (0.0)."

	if response_score >= 0.4:
		if ticket.category == "technical":
			return "Strong response with clear troubleshooting steps (0.4)."
		if ticket.category == "refund":
			return "Strong response with clear refund handling and process details (0.4)."
		return "Strong response with issue-specific handling and actionable detail (0.4)."
	if response_score >= 0.2:
		if ticket.category == "refund":
			return "Response partially addresses refund but lacks clear next steps (0.2)."
		return "Response is partially relevant but missing actionable detail (0.2)."
	if response_score >= 0.1:
		return "Response is generic and lacks actionable detail (0.1)."
	return "Response is irrelevant or too generic for this ticket category."

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Ticket(BaseModel):
	id: int
	content: str
	category: Literal["delivery", "refund", "technical", "general"]
	priority: Literal["low", "medium", "high"]


class PublicTicket(BaseModel):
	id: int
	content: str
	priority: Literal["low", "medium", "high"]


class Action(BaseModel):
	action_type: Literal["classify", "respond", "escalate"]
	category: Optional[Literal["delivery", "refund", "technical", "general"]] = None
	response: Optional[str] = None
	escalate: Optional[bool] = None


class Observation(BaseModel):
	current_ticket: PublicTicket
	history: List[Action] = Field(default_factory=list)


class Reward(BaseModel):
	score: float
	reason: str

	@field_validator("score")
	@classmethod
	def validate_score(cls, value: float) -> float:
		if value < 0.0:
			return 0.0
		if value > 1.0:
			return 1.0
		return value

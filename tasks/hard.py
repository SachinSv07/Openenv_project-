from graders.hard_grader import grade_hard
from models import Action, Ticket

TASK_NAME = "hard"
TASK_DESCRIPTION = "Classify, respond, and escalate correctly when priority is high."


def evaluate(ticket: Ticket, action: Action) -> float:
	return grade_hard(ticket, action)

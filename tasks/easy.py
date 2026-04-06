from graders.easy_grader import grade_easy
from models import Action, Ticket

TASK_NAME = "easy"
TASK_DESCRIPTION = "Classify the ticket into the correct category."


def evaluate(ticket: Ticket, action: Action) -> float:
	return grade_easy(ticket, action)

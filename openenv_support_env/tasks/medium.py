from graders.medium_grader import grade_medium
from models import Action, Ticket

TASK_NAME = "medium"
TASK_DESCRIPTION = "Classify the ticket and provide an appropriate response."


def evaluate(ticket: Ticket, action: Action) -> float:
    return grade_medium(ticket, action)

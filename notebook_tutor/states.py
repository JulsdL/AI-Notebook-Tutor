from typing import List, TypedDict
from langchain_core.messages import BaseMessage

# Define the state for the system
class TutorState(TypedDict):
    """
    A class representing the state of the tutor system.

    Attributes:
        messages (List[BaseMessage]): A list of messages in the system.
        next (str): The next step in the tutor system.
        quiz (List[dict]): A list of quiz questions and answers.
        quiz_created (bool): Indicates if a quiz has been created.
        question_answered (bool): Indicates if a question has been answered.
        flashcards_created (bool): Indicates if flashcards have been created.
    """
    messages: List[BaseMessage]
    next: str
    quiz: List[dict]
    quiz_created: bool
    question_answered: bool
    flashcards_created: bool

from typing import List, TypedDict
from langchain_core.messages import BaseMessage

# Define the state for the system
class TutorState(TypedDict):
    messages: List[BaseMessage]
    next: str
    quiz: List[dict]
    quiz_created: bool
    question_answered: bool
    flashcards_created: bool
    # flashcard_path: str

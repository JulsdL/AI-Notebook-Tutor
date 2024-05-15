from typing import Optional, Type
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
import csv
import uuid
import os

class FlashcardInput(BaseModel):
    flashcards: list = Field(description="A list of flashcards. Each flashcard should be a dictionary with 'question' and 'answer' keys.")

class FlashcardTool(BaseTool):
    name = "create_flashcards"
    description = "Create flashcards in a .csv format suitable for import into Anki"
    args_schema: Type[BaseModel] = FlashcardInput

    def _run(
        self, flashcards: list, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool to create flashcards."""
        filename = f"flashcards_{uuid.uuid4()}.csv"
        save_path = os.path.join('flashcards', filename)  # Save in 'flashcards' directory
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', newline='') as csvfile:
            fieldnames = ['Front', 'Back']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for card in flashcards:
                writer.writerow({'Front': card['question'], 'Back': card['answer']})
        return save_path

    async def _arun(
        self, flashcards: list, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("create_flashcards does not support async")

# Instantiate the tool
create_flashcards_tool = FlashcardTool()

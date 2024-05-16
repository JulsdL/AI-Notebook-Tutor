from typing import Optional, Type, Annotated
from pydantic import BaseModel, Field
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
        save_path = os.path.join('flashcards', filename)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'w', newline='') as csvfile:
            fieldnames = ['Front', 'Back']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for card in flashcards:
                writer.writerow({'Front': card['question'], 'Back': card['answer']})

        print("\033[93m" + f"Flashcards successfully created and saved to {save_path}" + "\033[0m")

        return "csv file created successfully."

    async def _arun(
        self, flashcards: list, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("create_flashcards does not support async")

# Instantiate the tool
create_flashcards_tool = FlashcardTool()

class RetrievalChainWrapper:
    def __init__(self, retrieval_chain):
        self.retrieval_chain = retrieval_chain

    def retrieve_information(
        self,
        query: Annotated[str, "query to ask the RAG tool"]
    ):
        """Use this tool to retrieve information about the provided notebook."""
        response = self.retrieval_chain.invoke({"question": query})
        return response["response"].content

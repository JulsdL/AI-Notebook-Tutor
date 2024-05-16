import os
import logging
import chainlit as cl
from dotenv import load_dotenv
from document_processing import DocumentManager
from retrieval import RetrievalManager
from langchain_core.messages import AIMessage, HumanMessage
from graph import create_tutor_chain, TutorState
import shutil

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

@cl.on_chat_start
async def start_chat():
    settings = {
        "model": "gpt4o",
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    cl.user_session.set("settings", settings)
    welcome_message = "Welcome to the Notebook-Tutor!"
    await cl.Message(content=welcome_message).send()

    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a Jupyter notebook (.ipynb, max. 5mb) to start:",
            accept={"application/x-ipynb+json": [".ipynb"]},
            max_size_mb=5
        ).send()

    file = files[0]  # Get the first file

    if file:
        notebook_path = file.path
        doc_manager = DocumentManager(notebook_path)
        doc_manager.load_document()
        doc_manager.initialize_retriever()
        cl.user_session.set("docs", doc_manager.get_documents())
        cl.user_session.set("retrieval_manager", RetrievalManager(doc_manager.get_retriever()))

        # Initialize LangGraph chain with the retrieval chain
        retrieval_chain = cl.user_session.get("retrieval_manager").get_RAG_QA_chain()
        cl.user_session.set("retrieval_chain", retrieval_chain)
        tutor_chain = create_tutor_chain(retrieval_chain)
        cl.user_session.set("tutor_chain", tutor_chain)

        ready_to_chat_message = "Notebook uploaded and processed successfully. You are now ready to chat!"
        await cl.Message(content=ready_to_chat_message).send()

        logger.info("Chat started and notebook uploaded successfully.")

@cl.on_message
async def main(message: cl.Message):

    # Retrieve the LangGraph chain from the session
    tutor_chain = cl.user_session.get("tutor_chain")

    if not tutor_chain:
        await cl.Message(content="No document processing setup found. Please upload a Jupyter notebook first.").send()
        return

    # Create the initial state with the user message
    user_message = message.content
    state = TutorState(
        messages=[HumanMessage(content=user_message)],
        next="supervisor",
        quiz=[],
        quiz_created=False,
        question_answered=False,
        flashcards_created=False,
    )

    logger.info(f"Initial state: {state}")

    # Process the message through the LangGraph chain
    for s in tutor_chain.stream(state, {"recursion_limit": 10}):
        logger.info(f"State after processing: {s}")

        agent_state = next(iter(s.values()))

        if "QAAgent" in s:
            if s['QAAgent']['question_answered']:
                qa_message = agent_state["messages"][-1].content
                logger.info(f"Sending QAAgent message: {qa_message}")
                await cl.Message(content=qa_message).send()

        if "QuizAgent" in s:
            if s['QuizAgent']['quiz_created']:
                quiz_message = agent_state["messages"][-1].content
                logger.info(f"Sending QuizAgent message: {quiz_message}")
                await cl.Message(content=quiz_message).send()

        if "FlashcardsAgent" in s:
            if s['FlashcardsAgent']['flashcards_created']:
                flashcards_message = agent_state["messages"][-1].content
                logger.info(f"Sending FlashcardsAgent message: {flashcards_message}")
                await cl.Message(content=flashcards_message).send()

                # Search for the flashcard file in the specified directory
                flashcard_directory = 'flashcards'
                flashcard_file = None
                latest_time = 0
                for root, dirs, files in os.walk(flashcard_directory):
                    for file in files:
                        if file.startswith('flashcards_') and file.endswith('.csv'):
                            file_path = os.path.join(root, file)
                            file_time = os.path.getmtime(file_path)
                            if file_time > latest_time:
                                latest_time = file_time
                                flashcard_file = file_path

                if flashcard_file:
                    logger.info(f"Flashcard path: {flashcard_file}")
                    # Use the File class to send the file
                    file_element = cl.File(name="Flashcards", path=flashcard_file, display="inline")
                    logger.info(f"Sending flashcards file: {file_element}")

                    await cl.Message(
                        content="Download the flashcards in .csv here:",
                        elements=[file_element]
                    ).send()

    logger.info("Reached END state.")


# @cl.on_chat_end
# async def end_chat():
#     # Clean up the flashcards directory
#     flashcard_directory = 'flashcards'
#     if os.path.exists(flashcard_directory):
#         shutil.rmtree(flashcard_directory)
#         os.makedirs(flashcard_directory)

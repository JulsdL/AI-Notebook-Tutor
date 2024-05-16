import os
import logging
import chainlit as cl
from dotenv import load_dotenv
from document_processing import DocumentManager
from retrieval import RetrievalManager
from langchain_core.messages import AIMessage, HumanMessage
from graph import create_tutor_chain, TutorState

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

@cl.on_chat_start
async def start_chat():
    settings = {
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    cl.user_session.set("settings", settings)
    welcome_message = "Welcome to the Notebook-Tutor! Please upload a Jupyter notebook (.ipynb and max. 5mb) to start."
    await cl.Message(content=welcome_message).send()

    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a Jupyter notebook (.ipynb, max. 5mb):",
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
        flashcard_filename="",
    )

    print("\033[93m" + f"Initial state: {state}" + "\033[0m")

    # Process the message through the LangGraph chain
    for s in tutor_chain.stream(state, {"recursion_limit": 3}):
        print("\033[93m" + f"State after processing: {s}" + "\033[0m")

        agent_state = next(iter(s.values()))
        print("\033[93m" + f"Agent state: {agent_state}" + "\033[0m")

        if "QAAgent" in s:
            if s['QAAgent']['question_answered']:
                print("\033[93m" + "************************Question answered**********************." + "\033[0m")
                qa_message = agent_state["messages"][-1].content
                await cl.Message(content=qa_message).send()

        if "QuizAgent" in s:
            if s['QuizAgent']['quiz_created']:
                print("\033[93m" + "************************Quiz created**********************." + "\033[0m")
                quiz_message = agent_state["messages"][-1].content
                await cl.Message(content=quiz_message).send()

        if "FlashcardsAgent" in s:
            if s['FlashcardsAgent']['flashcards_created']:
                print("\033[93m" + "************************Flashcards created**********************." + "\033[0m")
                flashcards_message = agent_state["messages"][-1].content
                await cl.Message(content=flashcards_message).send()

                flashcard_path = agent_state["flashcard_path"]
                print("\033[93m" + f"Flashcard path: {flashcard_path}" + "\033[0m")


                # Use the File class to send the file
                file_element = cl.File(name="Flashcards", path=flashcard_path)
                print("\033[93m" + f"Sending flashcards file: {file_element}" + "\033[0m")
                await cl.Message(
                    content="Here are your flashcards:",
                    elements=[file_element]
                ).send()

        final_state = s  # Save the final state after processing
        print("\033[93m" + f"Final state: {final_state}" + "\033[0m")

    print("\033[93m" + "Reached END state." + "\033[0m")

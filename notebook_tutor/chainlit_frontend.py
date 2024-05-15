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
    for s in tutor_chain.stream(state, {"recursion_limit": 10}):
        print("\033[93m" + f"State after processing: {s}" + "\033[0m")

        # Extract messages from the state
        if "__end__" not in s:
            agent_state = next(iter(s.values()))
            if "messages" in agent_state:
                response = agent_state["messages"][-1].content
                print("\033[93m" + f"Response: {response}" + "\033[0m")
                await cl.Message(content=response).send()
            else:
                print("Error: No messages found in agent state.")
        else:
            # Extract the final state
            final_state = next(iter(s.values()))
            print("\033[93m" + f"Final state: {final_state}" + "\033[0m")

            # Check if the quiz was created and send it to the frontend
            if final_state.get("quiz_created"):
                quiz_message = final_state["messages"][-1].content
                await cl.Message(content=quiz_message).send()

            # Check if a question was answered and send the response to the frontend
            if final_state.get("question_answered"):
                qa_message = final_state["messages"][-1].content
                await cl.Message(content=qa_message).send()

            # Check if flashcards are ready and send the file to the frontend
            if final_state.get("flashcards_created"):
                flashcards_message = final_state["messages"][-1].content
                await cl.Message(content=flashcards_message).send()

                # Create a relative path to the file
                flashcard_filename = final_state["flashcard_filename"]
                print("\033[93m" + f"Flashcard filename: {flashcard_filename}" + "\033[0m")
                flashcard_path = os.path.join(".files", flashcard_filename)
                print("\033[93m" + f"Flashcard path: {flashcard_path}" + "\033[0m")

                # Use the File class to send the file
                file_element = cl.File(name=os.path.basename(flashcard_path), path=flashcard_path)
                print("\033[93m" + f"Sending flashcards file: {file_element}" + "\033[0m")
                await cl.Message(
                    content="Here are your flashcards:",
                    elements=[file_element]
                ).send()

            print("\033[93m" + "Reached END state." + "\033[0m")

            break

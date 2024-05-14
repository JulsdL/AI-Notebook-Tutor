import chainlit as cl
from dotenv import load_dotenv
from document_processing import DocumentManager
from retrieval import RetrievalManager
from langchain_core.messages import AIMessage, HumanMessage
from graph import create_aims_chain, AIMSState

# Load environment variables
load_dotenv()

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
    welcome_message = "Welcome to the AIMS-Tutor! Please upload a Jupyter notebook (.ipynb and max. 5mb) to start."
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
        cl.user_session.set("retrieval_chain", retrieval_chain)  # Store the retrieval chain in the session
        aims_chain = create_aims_chain(retrieval_chain)
        cl.user_session.set("aims_chain", aims_chain)

@cl.on_message
async def main(message: cl.Message):
    # Retrieve the LangGraph chain from the session
    aims_chain = cl.user_session.get("aims_chain")

    if not aims_chain:
        await cl.Message(content="No document processing setup found. Please upload a Jupyter notebook first.").send()
        return

    # Create the initial state with the user message
    user_message = message.content
    state = AIMSState(messages=[HumanMessage(content=user_message)], next="supervisor", quiz=[])

    print(f"Initial state: {state}")

    # Process the message through the LangGraph chain
    for s in aims_chain.stream(state, {"recursion_limit": 10}):
        print(f"State after processing: {s}")

        # Extract messages from the state
        if "__end__" not in s:
            agent_state = next(iter(s.values()))
            if "messages" in agent_state:
                response = agent_state["messages"][-1].content
                print(f"Response: {response}")
                await cl.Message(content=response).send()
            else:
                print("Error: No messages found in agent state.")
        else:
            print("Reached end state.")
            break

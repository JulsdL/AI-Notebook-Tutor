import chainlit as cl
from dotenv import load_dotenv
from document_processing import DocumentManager
from retrieval import RetrievalManager

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

@cl.on_message
async def main(message: cl.Message):
    # Retrieve the multi-query retriever from session
    retrieval_manager = cl.user_session.get("retrieval_manager")
    if not retrieval_manager:
        await cl.Message(content="No document processing setup found. Please upload a Jupyter notebook first.").send()
        return

    question = message.content
    response = retrieval_manager.notebook_QA(question)  # Process the question

    msg = cl.Message(content=response)
    await msg.send()

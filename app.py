import os
from operator import itemgetter

import chainlit as cl
import tiktoken
from dotenv import load_dotenv


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyMuPDFLoader, PythonLoader, NotebookLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Configuration for OpenAI
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
openai_chat_model = ChatOpenAI(model="gpt-4-turbo", temperature=0)

# Define the RAG prompt
RAG_PROMPT = """
CONTEXT:
{context}

QUERY:
{question}

Answer the query in a pretty format if the context is related to it; otherwise, answer: 'Sorry, I can't answer.'
"""
rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)


# ChainLit setup for chat interaction
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

    # Display a welcoming message with instructions
    welcome_message = "Welcome to the AIMS-Tutor! Please upload a Jupyter notebook (.ipynb and max. 5mb) to start."
    await cl.Message(content=welcome_message).send()

    # Wait for the user to upload a file
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a Jupyter notebook (.ipynb, max. 5mb):",
            accept={"application/x-ipynb+json": [".ipynb"]},
            max_size_mb=5
        ).send()

    file = files[0] # Get the first file

    if file:
        # Load the Jupyter notebook
        notebook_path = file.path # Extract the path from the AskFileResponse object

        loader = NotebookLoader(
            notebook_path,
            include_outputs=True,
            max_output_length=20,
            remove_newline=True,
            traceback=False
        )
        docs = loader.load()
        cl.user_session.set("docs", docs) # Store the docs in the user session

        # Initialize the retriever components after loading document
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50, length_function=tiktoken_len) # Initialize the text splitter
        split_chunks = text_splitter.split_documents(docs) # Split the documents into chunks
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small") # Initialize the embedding model
        qdrant_vectorstore = Qdrant.from_documents(split_chunks, embedding_model, location=":memory:", collection_name="Notebook") # Create a Qdrant vector store
        qdrant_retriever = qdrant_vectorstore.as_retriever() # Set the Qdrant vector store as a retriever
        multiquery_retriever = MultiQueryRetriever.from_llm(retriever=qdrant_retriever, llm=openai_chat_model) # Create a multi-query retriever on top of the Qdrant retriever

        # Store the multiquery_retriever in the user session
        cl.user_session.set("multiquery_retriever", multiquery_retriever)

@cl.on_message
async def main(message: cl.Message):
    # Retrieve the multi-query retriever from session
    multiquery_retriever = cl.user_session.get("multiquery_retriever")

    if not multiquery_retriever:
        await message.reply("No document processing chain found. Please upload a Jupyter notebook first.")
        return

    question = message.content
    response = handle_query(question, multiquery_retriever)  # Process the question

    msg = cl.Message(content=response)
    await msg.send()


@cl.on_message
async def main(message: cl.Message):
    # Retrieve the multi-query retriever from session
    multiquery_retriever = cl.user_session.get("multiquery_retriever")
    if not multiquery_retriever:
        await message.reply("No document processing setup found. Please upload a Jupyter notebook first.")
        return

    question = message.content
    response = handle_query(question, multiquery_retriever)  # Process the question

    msg = cl.Message(content=response)
    await msg.send()

def handle_query(question, retriever):
    # Define the retrieval augmented query-answering chain
    retrieval_augmented_qa_chain = (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | openai_chat_model, "context": itemgetter("context")}
    )
    response = retrieval_augmented_qa_chain.invoke({"question": question})
    return response["response"].content

# Tokenization function
def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-3.5-turbo").encode(text)
    return len(tokens)

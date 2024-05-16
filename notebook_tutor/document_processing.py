import os
from langchain_community.document_loaders import NotebookLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain.retrievers import MultiQueryRetriever
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from notebook_tutor.utils import tiktoken_len

# Load environment variables
load_dotenv()

# Configuration for OpenAI
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
openai_chat_model = ChatOpenAI(model="gpt-4o", temperature=0.1)

class DocumentManager:
    """
    A class for managing documents and retrieving information from them.

    Attributes:
        notebook_path (str): The path to the notebook file.
        docs (list): A list of loaded documents.
        retriever (object): The retriever object used for document retrieval.

    Methods:
        load_document(): Loads the documents from the notebook file.
        initialize_retriever(): Initializes the retriever object for document retrieval.
        get_retriever(): Returns the retriever object.
        get_documents(): Returns the loaded documents.
    """
    def __init__(self, notebook_path):
        self.notebook_path = notebook_path
        self.docs = None
        self.retriever = None

    def load_document(self):
        """
        Loads the documents from the notebook file.

        This method initializes a `NotebookLoader` object with the specified parameters and uses it to load the documents from the notebook file. The loaded documents are stored in the `docs` attribute of the `DocumentManager` instance.

        Parameters:
            None

        Returns:
            None

        Raises:
            None
        """
        loader = NotebookLoader(
            self.notebook_path,
            include_outputs=False,
            max_output_length=20,
            remove_newline=True,
            traceback=False
        )
        self.docs = loader.load()

    def initialize_retriever(self):
        """
        A class for managing documents and retrieving information from them.

        Attributes:
            notebook_path (str): The path to the notebook file.
            docs (list): A list of loaded documents.
            retriever (object): The retriever object used for document retrieval.

        Methods:
            load_document(): Loads the documents from the notebook file.
            initialize_retriever(): Initializes the retriever object for document retrieval.
            get_retriever(): Returns the retriever object.
            get_documents(): Returns the loaded documents.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50, length_function=tiktoken_len)

        split_chunks = text_splitter.split_documents(self.docs)

        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

        qdrant_vectorstore = Qdrant.from_documents(split_chunks, embedding_model, location=":memory:", collection_name="Notebook")

        qdrant_retriever = qdrant_vectorstore.as_retriever()

        multiquery_retriever = MultiQueryRetriever.from_llm(retriever=qdrant_retriever, llm=openai_chat_model, include_original=True) # Create a multi-query retriever on top of the Qdrant retriever

        self.retriever = multiquery_retriever

    def get_retriever(self):
        return self.retriever

    def get_documents(self):
        return self.docs

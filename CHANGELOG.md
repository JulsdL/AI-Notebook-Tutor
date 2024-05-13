version 0.1.1 [2024-05-13]

## Modified

- Modularization: The code has been broken down into several modules, each with a specific responsibility. This makes the code easier to understand, test, and maintain. For example, the DocumentManager class in document_processing.py is responsible for managing documents and retrieving information from them. Similarly, the RetrievalManager class in retrieval.py is responsible for processing questions using a retrieval-augmented QA chain and returning the response.

- Separation of Concerns: The frontend and backend logic have been separated into different files (chainlit_frontend.py and document_processing.py, retrieval.py, etc.), which makes the codebase easier to navigate and maintain.

- Encapsulation: The code now makes use of classes and methods to encapsulate related functionality. For instance, the DocumentManager class encapsulates the functionality related to document management, and the RetrievalManager class encapsulates the functionality related to question processing and response retrieval.

version 0.1.0 [2024-05-13]

## Added

- Introduced a RAG application for QA in a Jupyter Notebook, enhancing the project's capabilities for document processing and query answering.
- Implemented Chainlit's `chainlit` Python package to support the RAG application's integration with the Chainlit platform.
- Added a new `app.py` script, establishing the core functionality for the RAG application, including document loading, retrieval QA chain, query processing, and user interaction.
- Updated project configuration and dependencies to support the new RAG application features.

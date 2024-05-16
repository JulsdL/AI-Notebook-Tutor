version 0.3.0 [2024-05-16]

## Added

- Implemented the Flashcards feature, allowing users to generate and download flashcards based on Jupyter notebook content.
- Introduced new agents (QAAgent, QuizAgent, FlashcardsAgent) for handling specific tasks like question answering, quiz generation, and flashcard creation.
- Added a new `chainlit_frontend.py` module for Chainlit integration, enabling interactive chat functionality with users.
- Developed a comprehensive state management system (`states.py`) to track conversation states and user interactions.
- Created `prompt_templates.py` and `graph.py` modules to define prompt templates and the conversation flow graph, respectively.
- Established a new tool (`tools.py`) for flashcard creation, supporting CSV format suitable for Anki import.

## Modified

- Updated the README.md to reflect the new project name (AI-Notebook-Tutor) and provided updated instructions for running the application.
- Enhanced the `.gitignore` file to include directories for flashcards and Chainlit configurations, improving project organization.

version 0.2.0 [2024-05-14]

## Added

- Introduced a comprehensive quiz functionality with LangGraph integration, enabling dynamic quiz generation and question answering based on Jupyter notebook content.
- Added new Python dependencies (`langgraph==0.0.48`) to support the quiz functionality and improved interaction flow.
- Implemented a new `graph.py` module to define the quiz and QA agents, along with the supervisor logic for managing conversation flow between agents.
- Enhanced the `chainlit_frontend.py` to integrate the LangGraph chain, facilitating real-time interaction with the quiz and QA functionality.
- Updated the `document_processing.py` and `retrieval.py` modules to support the new quiz functionality, including adjustments to the OpenAI model configuration and retrieval logic.

## Modified

- Updated the OpenAI model used in `document_processing.py` from "gpt-4-turbo" to "gpt-4o" to improve the quality of document processing and retrieval.
- Refined the retrieval logic in `retrieval.py` to include a new method for initializing the RAG QA chain, enhancing the system's ability to provide accurate and contextually relevant answers.

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

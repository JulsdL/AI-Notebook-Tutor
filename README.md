# AIMS-Tutor

# RAG Application for QA in Jupyter Notebook

AIMS-Tutor is designed to provide question-answering capabilities in a Jupyter Notebook using the Retrieval Augmented Generation (RAG) model. It's built on top of the LangChain and Chainlit platforms, and it uses the OpenAI API for the chat model.

## Features

- Document processing: Load a Jupyter notebook and split it into chunks for processing.
- Query answering: Use the RAG model to answer queries based on the processed document.
- User interaction: Interact with the application through a chat interface.

## Setup

1. Clone the repository.
2. Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

3. Create a .env file and add you environment variables. You'll need to provide your OpenAI API key in the format:

```bash
OPENAI_API_KEY=your-key-here
```

4. Run the application using the following command:

```bash
chainlit run aims_tutor/app.py
```

## Usage

Start a chat session and upload a Jupyter notebook file. The application will process the document and you can then ask questions related to the content of the notebook. It might take some time to answer some question (should be less than 1 min), so please be patient.

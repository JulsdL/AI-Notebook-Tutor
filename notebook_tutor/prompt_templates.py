from langchain_core.prompts import ChatPromptTemplate

class PromptTemplates:
    """
    The PromptTemplates class represents a collection of prompt templates used for generating chat prompts.

    Attributes:
        rag_QA_prompt (ChatPromptTemplate): A prompt template for generating RAG QA prompts.

    Methods:
        __init__(): Initializes all prompt templates as instance variables.
        get_rag_qa_prompt(): Returns the RAG QA prompt.

    Example usage:
        prompt_templates = PromptTemplates()
        rag_qa_prompt = prompt_templates.get_rag_qa_prompt()
    """
    def __init__(self):
        # Initializes all prompt templates as instance variables
        self.rag_QA_prompt = ChatPromptTemplate.from_template("""
            CONTEXT:
            {context}

            QUERY:
            {question}

            Answer the query in a pretty format if the context is related to it; otherwise, answer: 'Sorry, I can't answer. Please ask another question.'
        """)

    def get_rag_qa_prompt(self):
        # Returns the RAG QA prompt
        return self.rag_QA_prompt

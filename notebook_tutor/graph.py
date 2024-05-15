from typing import Annotated
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from tools import create_flashcards_tool
from states import TutorState
import functools

# Load environment variables
load_dotenv()

# Instantiate the language model
llm = ChatOpenAI(model="gpt-4o")

class RetrievalChainWrapper:
    def __init__(self, retrieval_chain):
        self.retrieval_chain = retrieval_chain

    def retrieve_information(
        self,
        query: Annotated[str, "query to ask the RAG tool"]
    ):
        """Use this tool to retrieve information about the provided notebook."""
        response = self.retrieval_chain.invoke({"question": query})
        return response["response"].content

# Create an instance of the wrapper
def get_retrieve_information_tool(retrieval_chain):
    wrapper_instance = RetrievalChainWrapper(retrieval_chain)
    return tool(wrapper_instance.retrieve_information)

# Instantiate the tools
flashcard_tool = create_flashcards_tool

# Function to create agents
def create_agent(
    llm: ChatOpenAI,
    tools: list,
    system_prompt: str,
) -> AgentExecutor:
    """Create a function-calling agent and add it to the graph."""
    system_prompt += "\nWork autonomously according to your specialty, using the tools available to you."
    " Do not ask for clarification."
    " Your other team members (and other teams) will collaborate with you with their own specialties."
    " You are chosen for a reason! You are one of the following team members: {team_members}."
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_functions_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)
    return executor

# Function to create agent nodes
def agent_node(state, agent, name):
    result = agent.invoke(state)
    if 'messages' not in result:
        raise ValueError(f"No messages found in agent state: {result}")
    new_state = {"messages": state["messages"] + [AIMessage(content=result["output"], name=name)]}

    # Set the appropriate flags and next state
    if name == "QuizAgent":
        new_state["quiz_created"] = True
    elif name == "QAAgent":
        new_state["question_answered"] = True
    elif name == "FlashcardsAgent":
        new_state["flashcards_created"] = True
        new_state["flashcard_filename"] = result["output"].split('(')[-1].strip(')')

    new_state["next"] = "FINISH"
    return new_state



# Function to create the supervisor
def create_team_supervisor(llm: ChatOpenAI, system_prompt, members) -> AgentExecutor:
    """An LLM-based router."""
    options = ["WAIT", "FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                },
            },
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we WAIT for user input? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), team_members=", ".join(members))
    return (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )


# Create the LangGraph chain
def create_tutor_chain(retrieval_chain):

    retrieve_information_tool = get_retrieve_information_tool(retrieval_chain)

    # Create QA Agent
    qa_agent = create_agent(
        llm,
        [retrieve_information_tool],
        "You are a QA assistant who answers questions about the provided notebook content.",
    )

    qa_node = functools.partial(agent_node, agent=qa_agent, name="QAAgent")

    # Create Quiz Agent
    quiz_agent = create_agent(
        llm,
        [retrieve_information_tool],
        """You are a quiz creator that generates quizzes based on the provided notebook content.
        First, You MUST Use the retrieval_inforation_tool to gather context from the notebook to gather relevant and accurate information.
        Next, create a 5-question quiz based on the information you have gathered. Include the answers at the end of the quiz.
        Present the quiz to the user in a clear and concise manner."""
    )

    quiz_node = functools.partial(agent_node, agent=quiz_agent, name="QuizAgent")

    # Create Flashcards Agent
    flashcards_agent = create_agent(
        llm,
        [retrieve_information_tool, flashcard_tool],
        """
        You are the Flashcard creator. Your mission is to create effective and concise flashcards based on the user's query and the content of the provided notebook. Your role involves the following tasks:
        1. Analyze User Query: Understand the user's request and determine the key concepts and information they need to learn.
        2. Search Notebook Content: Use the notebook content to gather relevant information and generate accurate and informative flashcards.
        3. Generate Flashcards: Create a series of flashcards content with clear questions on the front and detailed answers on the back. Ensure that the flashcards cover the essential points and concepts requested by the user.
        4. Export Flashcards: Use the flashcard_tool to create and export the flashcards in a format that can be easily imported into a flashcard management system, such as Anki.

        Remember, your goal is to help the user learn efficiently and effectively by breaking down the notebook content into manageable, repeatable flashcards."""
    )

    flashcards_node = functools.partial(agent_node, agent=flashcards_agent, name="FlashcardsAgent")

    # Create Supervisor Agent
    supervisor_agent = create_team_supervisor(
        llm,
        "You are a supervisor tasked with managing a conversation between the following agents: QAAgent, QuizAgent, FlashcardsAgent. Given the user request, decide which agent should act next.",
        ["QAAgent", "QuizAgent", "FlashcardsAgent"],
    )

    # Build the LangGraph
    tutor_graph = StateGraph(TutorState)
    tutor_graph.add_node("QAAgent", qa_node)
    tutor_graph.add_node("QuizAgent", quiz_node)
    tutor_graph.add_node("FlashcardsAgent", flashcards_node)
    tutor_graph.add_node("supervisor", supervisor_agent)

    tutor_graph.add_edge("QAAgent", "supervisor")
    tutor_graph.add_edge("QuizAgent", "supervisor")
    tutor_graph.add_edge("FlashcardsAgent", "supervisor")
    tutor_graph.add_conditional_edges(
        "supervisor",
        lambda x: "FINISH" if x.get("quiz_created") or x.get("question_answered") or x.get("flashcards_created") else x["next"],
        {"QAAgent": "QAAgent",
        "QuizAgent": "QuizAgent",
        "FlashcardsAgent": "FlashcardsAgent",
        "FINISH": END},
    )

    tutor_graph.set_entry_point("supervisor")
    return tutor_graph.compile()

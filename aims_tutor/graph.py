from typing import Annotated, List, TypedDict
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
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
    if 'messages' not in result: # Check if messages are present in the agent state
        raise ValueError(f"No messages found in agent state: {result}")
    new_state = {"messages": state["messages"] + [AIMessage(content=result["output"], name=name)]}
    if "next" in result:
        new_state["next"] = result["next"]
    if name == "QuizAgent" and "quiz_created" in state and not state["quiz_created"]:
        new_state["quiz_created"] = True
        new_state["next"] = "FINISH" # Finish the conversation after the quiz is created and wait for a new user input
    if name == "QAAgent":
        new_state["question_answered"] = True
        new_state["next"] = "question_answered"
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

# Define the state for the system
class AIMSState(TypedDict):
    messages: List[BaseMessage]
    next: str
    quiz: List[dict]
    quiz_created: bool
    question_answered: bool


# Create the LangGraph chain
def create_aims_chain(retrieval_chain):

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
        "You are a quiz creator that generates quizzes based on the provided notebook content."

        """First, You MUST Use the retrieval_inforation_tool to gather context from the notebook to gather relevant and accurate information.

        Next, create a 5-question quiz based on the information you have gathered. Include the answers at the end of the quiz.

        Present the quiz to the user in a clear and concise manner."""
    )

    quiz_node = functools.partial(agent_node, agent=quiz_agent, name="QuizAgent")

    # Create Supervisor Agent
    supervisor_agent = create_team_supervisor(
        llm,
        "You are a supervisor tasked with managing a conversation between the following agents: QAAgent, QuizAgent. Given the user request, decide which agent should act next.",
        ["QAAgent", "QuizAgent"],
    )

    # Build the LangGraph
    aims_graph = StateGraph(AIMSState)
    aims_graph.add_node("QAAgent", qa_node)
    aims_graph.add_node("QuizAgent", quiz_node)
    aims_graph.add_node("supervisor", supervisor_agent)

    aims_graph.add_edge("QAAgent", "supervisor")
    aims_graph.add_edge("QuizAgent", "supervisor")
    aims_graph.add_conditional_edges(
        "supervisor",
        lambda x: "FINISH" if x.get("quiz_created") else ("FINISH" if x.get("question_answered") else x["next"]),
        {"QAAgent": "QAAgent", "QuizAgent": "QuizAgent", "WAIT": END, "FINISH": END, "question_answered": END},
    )

    aims_graph.set_entry_point("supervisor")
    return aims_graph.compile()

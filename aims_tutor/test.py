from typing import Annotated, List, TypedDict
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
import functools

# Load environment variables
load_dotenv()

@tool
def generate_quiz(
    documents: Annotated[List[str], "List of documents to generate quiz from"],
    num_questions: Annotated[int, "Number of questions to generate"] = 5
) -> Annotated[List[dict], "List of quiz questions"]:
    """Generate a quiz based on the provided documents."""
    # Placeholder logic for quiz generation
    # In a real scenario, you'd use NLP techniques to generate questions
    questions = [{"question": f"Question {i+1}", "options": ["Option 1", "Option 2", "Option 3"], "answer": "Option 1"} for i in range(num_questions)]
    return questions

@tool
def retrieve_information(
    query: Annotated[str, "query to ask the retrieve information tool"]
    ):
    """Use Retrieval Augmented Generation to retrieve information about the provided content."""
    return {"response": "This is a placeholder response for retrieval information."}


# Define a function to create agents
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
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

# Define a function to create agent nodes
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": state["messages"] + [AIMessage(content=result["output"], name=name)]}

# Define a function to create the supervisor
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


# Instantiate the language model
llm = ChatOpenAI(model="gpt-4o")

# Create QA Agent
qa_agent = create_agent(
    llm,
    [retrieve_information],  # Existing QA tool
    "You are a QA assistant who answers questions about the provided notebook content.",
)
qa_node = functools.partial(agent_node, agent=qa_agent, name="QAAgent")

# Create Quiz Agent
quiz_agent = create_agent(
    llm,
    [generate_quiz],
    "You are a quiz creator that generates quizzes based on the provided notebook content.",
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
    lambda x: x["next"],
    {"QAAgent": "QAAgent", "QuizAgent": "QuizAgent", "WAIT": END, "FINISH": END},
)

aims_graph.set_entry_point("supervisor")
chain = aims_graph.compile()

if __name__ == "__main__":

    # Define the function to enter the chain
    def enter_chain(message: str):
        results = {
            "messages": [HumanMessage(content="I'd like to take a quiz based on the uploaded notebook.")],
        }
        return results

    aims_chain = enter_chain | chain

    for s in aims_chain.stream(
        "I'd like to take a quiz based on the uploaded notebook.", {"recursion_limit": 15}
    ):
        if "__end__" not in s:
            print(s)
            print("---")

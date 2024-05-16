from typing import Annotated
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_openai import ChatOpenAI
from tools import create_flashcards_tool, RetrievalChainWrapper


# Instantiate the language model
llm = ChatOpenAI(model="gpt-4o")

# Function to create an instance of the retrieval tool wrapper
def get_retrieve_information_tool(retrieval_chain):
    wrapper_instance = RetrievalChainWrapper(retrieval_chain)
    return tool(wrapper_instance.retrieve_information)

# Instantiate the flashcard tool
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

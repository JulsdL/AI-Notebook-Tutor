from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from states import TutorState
from agents import create_agent, agent_node, create_team_supervisor, get_retrieve_information_tool, llm, flashcard_tool
from prompt_templates import PromptTemplates
import functools

# Load environment variables
load_dotenv()

# Create the LangGraph chain
def create_tutor_chain(retrieval_chain):
    retrieve_information_tool = get_retrieve_information_tool(retrieval_chain)

    # Create QA Agent
    qa_agent = create_agent(
        llm,
        [retrieve_information_tool],
        PromptTemplates().get_qa_agent_prompt(),
    )
    qa_node = functools.partial(agent_node, agent=qa_agent, name="QAAgent")

    # Create Quiz Agent
    quiz_agent = create_agent(
        llm,
        [retrieve_information_tool],
        PromptTemplates().get_quiz_agent_prompt(),
    )
    quiz_node = functools.partial(agent_node, agent=quiz_agent, name="QuizAgent")

    # Create Flashcards Agent
    flashcards_agent = create_agent(
        llm,
        [retrieve_information_tool, flashcard_tool],
        PromptTemplates().get_flashcards_agent_prompt(),
    )
    flashcards_node = functools.partial(agent_node, agent=flashcards_agent, name="FlashcardsAgent")

    # Create Supervisor Agent
    supervisor_agent = create_team_supervisor(
        llm,
        PromptTemplates().get_supervisor_agent_prompt(),
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

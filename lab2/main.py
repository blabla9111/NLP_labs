import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import config
from langchain.agents.structured_output import ToolStrategy
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek.chat_models import ChatDeepSeek
from langgraph.types import Command

from langchain_core.messages import RemoveMessage
from typing import Literal

from lab2.agents.react_agent import ReActAgent
from lab2.agents.pinn_loss_agent import PINNLossWieghtsGenerator
from lab2.agents.weight_checker import WeightChecker
from lab2.agents.writer_agent import PINNResultsWriter
from lab2.agent_tools.comment_tools import check_comment_validity, get_new_comment_from_expert
from lab2.agent_tools.comment_classifier import get_class_subclass_names
from lab2.data_formats.input_output_formats import GraphState, ExpertComment, PINNLossWeights, WeightValidationResult

from langgraph.checkpoint.memory import InMemorySaver
import uuid


def need_retry_generator(state: GraphState) -> Command[Literal["PINNLossWeightsAgent", "WriterAgent"]]:
    print("START RetryChecker")
    validation_errors = state["validation_errors"]
    handoff_count = state["handoff_count"]
    # this is a replacement for a conditional edge function
    if validation_errors and handoff_count != 5:
        current_agent = "PINNLossWeightsAgent"
        goto = "PINNLossWeightsAgent"
    else:
        current_agent = "WriterAgent"
        goto = "WriterAgent"

    # note how Command allows you to BOTH update the graph state AND route to the next node
    return Command(
        # this is the state update
        update={"current_agent": current_agent},
        # this is a replacement for an edge
        goto=goto,
    )

def clear_some_messages(state: GraphState) -> GraphState:
    """Clear ALL messages from the conversation"""
    print("START CleanerNode")
    messages = state["messages"]
    if len(messages) > 2:
        # Create RemoveMessage for first two
        remove_instructions = [RemoveMessage(id=m.id) for m in messages[:2]]
        remaining_messages = messages[2:]
        new_messages = remove_instructions + remaining_messages
        
        return {"messages": new_messages}
    
    return state

llm = ChatDeepSeek(
    api_base=config.BASE_URL,
    base_url=config.BASE_URL,
    api_key=config.API_KEY,
    model=config.MODEL_NAME,
    streaming=False,
    timeout=120
)

react_agent = ReActAgent(model=llm,
                         tools = [check_comment_validity, get_new_comment_from_expert, get_class_subclass_names],
                         response_format=ToolStrategy(ExpertComment))


pinn_loss_weights_agent = PINNLossWieghtsGenerator(model=llm,
                                                   parser_output_class=PINNLossWeights)

checker_agent = WeightChecker(model=llm,
                              parser_output_class=WeightValidationResult)

writer_agent = PINNResultsWriter(default_filename="result")


builder = StateGraph(GraphState)
builder.add_node("ReActAgent", react_agent)
builder.add_node("CleanerNode", clear_some_messages)
builder.add_node("PINNLossWeightsAgent", pinn_loss_weights_agent)
builder.add_node("WeightCheckerAgent", checker_agent)
builder.add_node("RetryChecker", need_retry_generator)
builder.add_node("WriterAgent", writer_agent)

builder.add_edge(START, "ReActAgent")
builder.add_edge("ReActAgent", "CleanerNode")
builder.add_edge("CleanerNode", "PINNLossWeightsAgent")
builder.add_edge("PINNLossWeightsAgent", "WeightCheckerAgent")
builder.add_edge("WeightCheckerAgent", "RetryChecker")
builder.add_edge("WriterAgent", END)


checkpointer = InMemorySaver()

graph = builder.compile(checkpointer=checkpointer)
graph_png = graph.get_graph(xray=True)
png_bytes = graph_png.draw_mermaid_png()

with open("SciResearch_agent.png", "wb") as f:
    f.write(png_bytes)

if __name__ == "__main__":
    # "The symmetry of the infection curve is unrealistic - the descent should be slower than the ascent."
    
    # Создаем улучшенный системный промпт
    system_prompt = """You are an epidemiological forecast validation agent. 
                    Your task is to check expert_comment validity and then find class and subclass for expert comment.
                    If expert comment is not valid, ask user to write a new one, then check comment validity again.

                    ## OUTPUT REQUIREMENTS:
                    You MUST return the result in this EXACT JSON format corresponding to the ExpertComment model:
                    {
                        "comment": "The expert's comment text",
                        "is_valid": true/false,
                        "reason": "Explanation of validation decision",
                        "recommendations": "Suggestions for improvement if invalid",
                        "comment_class": "Class name from classification system",
                        "comment_subclass": "Subclass name from classification system"
                    }

                    ## VALIDATION CRITERIA:
                    1. CONCRETENESS: Comment should be specific and actionable
                    2. CERTAINTY: Should not be ambiguous or vague
                    3. OBJECTIVITY: Should be based on observable facts, not personal preferences

                    ## CLASSIFICATION REQUIREMENTS:
                    - Use the get_class_subclass_names function to determine the appropriate class and subclass
                    - The classification should match the content and intent of the expert comment

                    ## WORKFLOW:
                    1. First, validate the expert comment against the criteria (CONCRETENESS, CERTAINTY, OBJECTIVITY)
                    2. If valid → proceed to classification
                    3. If invalid → provide specific recommendations for improvement and ask for a revised comment
                    4. After user provides a revised comment, repeat validation

                    ## RESPONSE FORMAT:
                    Always respond with ONLY the valid JSON object in the specified format.
                    Never add extra explanations, questions, or markdown formatting outside the JSON.
                    The JSON must contain all six required fields exactly as defined above.

                    Remember: Respond ONLY with the JSON object, nothing else."""
    
    # session_id = str(uuid.uuid4())[:8]
    # configuration = {"configurable": {"thread_id": f"pinn_session_{session_id}"}}
    
    # First invocation
    # initial_state = {
    #     "messages": [
    #         SystemMessage(content=system_prompt),
    #         HumanMessage(content="I do not like. Curve should be smoother")
    #     ],
    #     "current_response": "",
    #     "expert_comment": None,
    #     "handoff_count": 0,
    #     "session_id": session_id
    # }
    
    # print("=== FIRST INVOCATION ===")
    # final_state1 = graph.invoke(initial_state, config=configuration)
    
    # print(f"Final agent after first run: {final_state1.get('current_agent', 'Not set')}")
    # print(f"Handoff count: {final_state1.get('handoff_count', 0)}")
    
    # # Second invocation with same config - should continue from saved state
    # print("\n=== SECOND INVOCATION (same config) ===")
    # new_input_state = {
    #     "messages": [HumanMessage(content="Here's an improved comment: The symmetry of the infection curve is unrealistic - the descent should be slower than the ascent.")],
    #     "handoff_count": final_state1.get("handoff_count", 0),
    #     "session_id": session_id
    # }
    
    # final_state2 = graph.invoke(new_input_state, config=configuration)
    # print(f"Final agent after second run: {final_state2.get('current_agent', 'Not set')}")
    # print(f"Handoff count: {final_state2.get('handoff_count', 0)}")
    
    # # Verify persistence
    # if final_state2.get('handoff_count', 0) > final_state1.get('handoff_count', 0):
    #     print("\n✓ State persistence is WORKING: Handoff count increased from saved state")
    # else:
    #     print("\n✗ State persistence NOT working: Handoff count didn't persist")
    # Создаем уникальный thread_id для этой сессии
    session_id = str(uuid.uuid4())[:8]
    configuration = {"configurable": {"thread_id": f"pinn_session_{session_id}"}}
    
    initial_state = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content="The symmetry of the infection curve is unrealistic - the descent should be slower than the ascent.")
        ],
        "current_response": "",
        "expert_comment": None,
        "handoff_count": 0,
        "session_id": session_id
    }
    final_state = graph.invoke(initial_state, config=configuration)
    print("Final state is")
    print(final_state['messages'])
    print("="*50)
    print(final_state['expert_comment'])
    print("="*50)
    print(final_state['loss_weights'])
    print("="*50)
    print(final_state['current_agent'])
    


# https://habr.com/ru/companies/amvera/articles/949376/ Создание умных AI-агентов: полный курс по LangGraph от А до Я.

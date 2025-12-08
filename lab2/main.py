import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import config
from langchain.agents.structured_output import ToolStrategy
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek.chat_models import ChatDeepSeek
from langgraph.types import Command
from typing import Literal

from lab2.agents.react_agent import ReActAgent
from lab2.agents.pinn_loss_agent import PINNLossWieghtsGenerator
from lab2.agents.weight_checker import WeightChecker
from lab2.agents.check_comment_agent import check_comment_validity, get_new_comment_from_expert
from lab2.agent_tools.comment_classifier import get_class_subclass_names
from lab2.data_formats.input_output_formats import GraphState, ExpertComment, PINNLossWeights, WeightValidationResult


def need_retry_generator(state: GraphState) -> Command[Literal["PINNLossWeightsAgent", END]]:
    print("need_retry_generator")
    validation_errors = state["validation_errors"]
    handoff_count = state["handoff_count"]
    # this is a replacement for a conditional edge function
    if validation_errors and handoff_count != 5:
        current_agent = "PINNLossWeightsAgent"
        goto = "PINNLossWeightsAgent"
    else:
        current_agent = "END"
        goto = END

    # note how Command allows you to BOTH update the graph state AND route to the next node
    return Command(
        # this is the state update
        update={"current_agent": current_agent},
        # this is a replacement for an edge
        goto=goto,
    )

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

# agent_node = AgentNode(model=llm,
#                        tools=[get_new_query_from_user, get_topic, do_research],
#                        response_format=ToolStrategy(ResultSummary))

# github_node = GitHubRepoSearcher(github_token=config.GITHUB_API_TOKEN, default_max_results=3)

# summary_node = SummaryGenerator(model=llm, parser_output_class=ResearchSummary)

# writer_node = ResearchReportWriter()


builder = StateGraph(GraphState)
builder.add_node("ReActAgentNode", react_agent)
builder.add_node("PINNLossWeightsAgent", pinn_loss_weights_agent)
builder.add_node("WeightCheckerAgent", checker_agent)
builder.add_node("RetryChecker", need_retry_generator)

builder.add_edge(START, "ReActAgentNode")
builder.add_edge("ReActAgentNode", "PINNLossWeightsAgent")
builder.add_edge("PINNLossWeightsAgent", "WeightCheckerAgent")
# builder.add_edge("GuthubNode", "WriterNode")
# builder.add_edge("SummaryNode", "WriterNode")
builder.add_edge("WeightCheckerAgent", "RetryChecker")

graph = builder.compile()
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
    initial_state = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content="I do not like. Curve should be smoother")
        ],
        "current_response": "",
        "expert_comment": None,
        "handoff_count": 0
    }
    final_state = graph.invoke(initial_state)
    print("Final state is")
    print(final_state['messages'])
    print("="*50)
    print(final_state['expert_comment'])
    print("="*50)
    print(final_state['loss_weights'])
    print("="*50)
    print(final_state['current_agent'])
    


# https://habr.com/ru/companies/amvera/articles/949376/ Создание умных AI-агентов: полный курс по LangGraph от А до Я.

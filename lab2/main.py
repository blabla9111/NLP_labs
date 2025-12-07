import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import config
from langchain.agents.structured_output import ToolStrategy
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek.chat_models import ChatDeepSeek

from lab2.agents.react_agent import ReActAgent
from lab2.agents.check_comment_agent import check_comment_validity, get_new_comment_from_expert
from lab2.agent_tools.comment_classifier import get_class_subclass_names
from lab2.data_formats.input_output_formats import GraphState, ExpertComment



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

# agent_node = AgentNode(model=llm,
#                        tools=[get_new_query_from_user, get_topic, do_research],
#                        response_format=ToolStrategy(ResultSummary))

# github_node = GitHubRepoSearcher(github_token=config.GITHUB_API_TOKEN, default_max_results=3)

# summary_node = SummaryGenerator(model=llm, parser_output_class=ResearchSummary)

# writer_node = ResearchReportWriter()


builder = StateGraph(GraphState)
builder.add_node("ReActAgentNode", react_agent)
# builder.add_node("SummaryNode", summary_node)
# builder.add_node("GuthubNode", github_node)
# builder.add_node("WriterNode", writer_node)

builder.add_edge(START, "ReActAgentNode")
# builder.add_edge("ReActAgentNode", "SummaryNode")
# builder.add_edge("ReActAgentNode", "GuthubNode")
# builder.add_edge("GuthubNode", "WriterNode")
# builder.add_edge("SummaryNode", "WriterNode")
builder.add_edge("ReActAgentNode", END)

graph = builder.compile()
graph_png = graph.get_graph(xray=True)
png_bytes = graph_png.draw_mermaid_png()

with open("SciResearch_agent.png", "wb") as f:
    f.write(png_bytes)

if __name__ == "__main__":
    # "The symmetry of the infection curve is unrealistic - the descent should be slower than the ascent."
    
    # Создаем улучшенный системный промпт
    system_prompt = """You are an epidemiological forecast validation agent. 
    Your task is check expert_comment validity and then find class and subclass for expert comment.


    Always respond in this exact format. Never add extra explanations or questions."""

    initial_state = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content="I do not like. Curve should be smoother")
        ],
        "current_response": "",
        "expert_comment": None
    }
    final_state = graph.invoke(initial_state)
    print("Final state is")
    print(final_state['messages'])
    print("="*50)
    print(final_state['expert_comment'])


# https://habr.com/ru/companies/amvera/articles/949376/ Создание умных AI-агентов: полный курс по LangGraph от А до Я.

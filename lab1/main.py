import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from input_output_formats import ResultSummary, ResearchSummary, GraphState
from lab1.agent_tools.do_research_tool import do_research
from lab1.agent_tools.new_query_tool import get_new_query_from_user
from lab1.agent_tools.topic_search_tool import get_topic
from lab1.agent_tools.arxiv_search_tool import arxiv_search
from current_date_tool import get_weather
import config
from langchain.agents.structured_output import ToolStrategy
from langgraph.graph import StateGraph, START, END
from typing import Any, Dict, List, Annotated, TypedDict
import operator
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek.chat_models import ChatDeepSeek



from lab1.nodes.agent_node import AgentNode
from lab1.nodes.summary_node import SummaryGenerator
from lab1.nodes.writer_node import ResearchReportWriter

# Add the parent directory to Python path



llm = ChatDeepSeek(
    api_base=config.BASE_URL,
    base_url=config.BASE_URL,
    api_key=config.API_KEY,
    model=config.MODEL_NAME,
    streaming=False,
    timeout=120
)

agent_node = AgentNode(model=llm,
                       tools=[get_new_query_from_user, get_topic, do_research],
                       response_format=ToolStrategy(ResultSummary))

summary_node = SummaryGenerator(model=llm, parser_output_class=ResearchSummary)

writer_node = ResearchReportWriter()


builder = StateGraph(GraphState)
builder.add_node("ReActAgentNode", agent_node)
builder.add_node("SummaryNode", summary_node)
builder.add_node("WriterNode", writer_node)

builder.add_edge(START, "ReActAgentNode")
builder.add_edge("ReActAgentNode", "SummaryNode")
builder.add_edge("SummaryNode", "WriterNode")
builder.add_edge("WriterNode", END)

graph = builder.compile()
graph_png = graph.get_graph(xray=True)
png_bytes = graph_png.draw_mermaid_png()

with open("simple_agent.png", "wb") as f:
    f.write(png_bytes)

if __name__ == "__main__":
    initial_state = {
        "messages": [
            SystemMessage(
                content="You are SciResearch Assistant, an expert AI agent specialized in finding and analyzing scientific information from arXiv and Crossref databases."),
            HumanMessage(content="EMJMCMC algorithms")
        ],
        # "intermediate_steps": [],
        "current_response": "",
        "result_summary": None
    }
    final_state = graph.invoke(initial_state)
    print("Final state is")
    # print(final_state['messages'])
    print("="*50)
    print(final_state['research_summary'])


# https://habr.com/ru/companies/amvera/articles/949376/ Создание умных AI-агентов: полный курс по LangGraph от А до Я.

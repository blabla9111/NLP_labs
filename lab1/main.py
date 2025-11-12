import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from langchain_deepseek.chat_models import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
import operator
from typing import Any, Dict, List, Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.agents.structured_output import ToolStrategy


import config
from current_date_tool import get_weather
from lab1.agent_tools.arxiv_search_tool import arxiv_search
from lab1.agent_tools.topic_search_tool import get_topic
from lab1.agent_tools.new_query_tool import get_new_query_from_user
from lab1.agent_tools.do_research_tool import do_research

from input_output_formats import ResultSummary

llm = ChatDeepSeek(
                        api_base=config.BASE_URL,
                        base_url=config.BASE_URL,
                        api_key=config.API_KEY,
                        model=config.MODEL_NAME,
                        streaming=False,
                        timeout=120
                        )

# agent = create_agent(model=llm, tools=[get_weather, arxiv_search, get_topic])

# chunks = agent.stream({
#     "messages":[
#         SystemMessage(content="You are a helpful assistant."),
#         HumanMessage(content="Hello! Find papers which is contain PINN in epidemiology")
#     ]
# })

# for chunk in chunks:
#     print(chunk)
#     print("--------")

class AgentState(TypedDict):
    messages: List  # Сообщения (SystemMessage, HumanMessage)
    intermediate_steps: Annotated[List[tuple], operator.add]  # Шаги выполнения
    current_response: str  # Текущий ответ

def agent_node(state: AgentState) -> AgentState:
    agent = create_agent(model=llm, tools=[get_new_query_from_user, get_topic, do_research], response_format=ToolStrategy(ResultSummary))
    chunks = agent.stream({
        "messages": state["messages"],
        "intermediate_steps": state["intermediate_steps"]
    })
    
    response_parts = []
    last_chunk = None
    structured_response = None
    
    for chunk in chunks:
        response_parts.append(str(chunk))
        last_chunk = chunk
        
        # Extract structured response if available
        if hasattr(chunk, 'model') and hasattr(chunk.model, 'structured_response'):
            structured_response = chunk.model.structured_response
        elif isinstance(chunk, dict) and chunk.get('model', {}).get('structured_response'):
            structured_response = chunk['model']['structured_response']
        
        print(chunk)
        print("--------")
    
    # Prepare the response for messages
    if structured_response:
        # Use the structured response content
        response_content = f"Topic: {structured_response.topic}\nSummary 1: {structured_response.arxiv_api_response}\nSummary 2: {structured_response.crossref_api_response}"
    else:
        # Fallback: convert chunks to string
        response_content = "\n".join(str(part) for part in response_parts)
    
    return {
        "current_response": structured_response,  # This will be None if no structured response found
        "messages": state["messages"] + [response_parts]
    }

builder = StateGraph(AgentState)
builder.add_node("ReActAgentNode", agent_node)

builder.add_edge(START, "ReActAgentNode")
builder.add_edge("ReActAgentNode", END)

graph = builder.compile()
graph_png = graph.get_graph(xray=True)
png_bytes = graph_png.draw_mermaid_png()

with open("simple_agent.png", "wb") as f:
    f.write(png_bytes)

if __name__ == "__main__":
    initial_state = {
        "messages": [
            SystemMessage(content="You are SciResearch Assistant, an expert AI agent specialized in finding and analyzing scientific information from arXiv and Crossref databases."),
            HumanMessage(content="Hello!  Find papers which contain PINN in epidemiology")
        ],
        "intermediate_steps": [],
        "current_response": ""
    }
    final_state = graph.invoke(initial_state)
    print("Final state is")
    print(final_state['current_response'])


# https://habr.com/ru/companies/amvera/articles/949376/ Создание умных AI-агентов: полный курс по LangGraph от А до Я.
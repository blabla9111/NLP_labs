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
from arxiv_search_tool import arxiv_search
from topic_search_tool import get_topic
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
    agent = create_agent(model=llm, tools=[get_weather, arxiv_search, get_topic], response_format=ToolStrategy(ResultSummary))
    chunks = agent.stream({
            "messages": state["messages"],
            "intermediate_steps": state["intermediate_steps"]
        })
        
    response_parts = []
    last_chunk = {}
    for chunk in chunks:
        response_parts.append(str(chunk))
        last_chunk = chunk
        # break
        print(type(chunks))
        print("--------")
        
    full_response = "".join(response_parts)

    # print(full_response)
        
    return {
            "current_response": last_chunk["model"]["structured_response"],
            "messages": state["messages"] + [HumanMessage(content=full_response)]
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
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello! Find papers which contain PINN in epidemiology")
        ],
        "intermediate_steps": [],
        "current_response": ""
    }
    final_state = graph.invoke(initial_state)
    print("Final state is")
    print(final_state['current_response'])


# https://habr.com/ru/companies/amvera/articles/949376/ Создание умных AI-агентов: полный курс по LangGraph от А до Я.
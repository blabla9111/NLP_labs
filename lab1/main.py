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

from input_output_formats import ResultSummary, ResearchSummary

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
    result_summary: ResultSummary
    research_summary: ResearchSummary


def agent_node(state: AgentState) -> AgentState:
    agent = create_agent(model=llm, tools=[get_new_query_from_user, get_topic, do_research], response_format=ToolStrategy(ResultSummary))
    chunks = agent.stream({
        "messages": state["messages"],
        # "intermediate_steps": state["intermediate_steps"]
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
        "messages": state["messages"] + [response_parts],
        "result_summary": structured_response
    }

def get_summary(state: AgentState) -> AgentState:
    print("START SUMMARIZING")
    topic = state["result_summary"].topic
    arxiv_papers_info = state["result_summary"].arxiv_api_response
    crossref_papers_info = state["result_summary"].crossref_api_response
    parser = PydanticOutputParser(pydantic_object=ResearchSummary)
    prompt = ChatPromptTemplate(messages=[
        ("system", """You are an expert research assistant specialized in analyzing and summarizing academic papers. Your task is to extract key information from research papers and return structured summaries.

        {format_instructions}

        Analyze the provided research papers and return a JSON object with the following structure:
        - trends_info: Identify major trends, research directions, and emerging themes in the topic area
        - methods: Extract and categorize the research methodologies, algorithms, and techniques used across the papers  
        - limitations: Document the limitations, disadvantages, and gaps mentioned in the papers

        Return ONLY valid JSON in this exact format. Do not include any additional text, explanations, or markdown formatting."""),
            
            ("human", """Please analyze the following research papers and provide a comprehensive summary:

        Research topic:
             {topic}
                  
        Research Papers Context:
        ArXiv Papers
             {arxiv_papers_info}
        
        CrossRef Papers
             {crossref_papers_info}

        Focus on extracting:
        1. TRENDS: Current research directions, popular approaches, and evolving themes in this field
        2. METHODS: Specific methodologies, algorithms, frameworks, and experimental approaches used
        3. LIMITATIONS: Explicitly stated limitations, methodological constraints, and areas needing improvement

        Return only the valid JSON object without any additional text.""")
        ],
        partial_variables={"format_instructions": parser.get_format_instructions()})
    
    # llm = ChatDeepSeek(
    #                     api_base=BASE_URL,
    #                     base_url=BASE_URL,
    #                     api_key=API_KEY,
    #                     model=MODEL_NAME,
    #                     streaming=False,
    #                     timeout=120
    #                     )
    
    chain = prompt | llm | parser

    output: ResearchSummary = chain.invoke({"topic":topic,
                                            "arxiv_papers_info": arxiv_papers_info,
                                            "crossref_papers_info": crossref_papers_info})
    print(output)
    # print()
    return {"research_summary": output}

def write_beautiful_answer_to_file(topic, research_summary, filename: str = "answer.txt") -> None:
    """
    Write research findings to a beautifully formatted text file using ResearchSummary object
    
    Args:
        research_summary: ResearchSummary object containing trends, methods, and limitations
        filename: Output filename
    """
    
    with open(filename, 'w', encoding='utf-8') as f:
        
        # Header
        f.write("=" * 80 + "\n")
        f.write("RESEARCH ANALYSIS REPORT\n")
        f.write(f"TOPIC: {topic}\n")
        f.write("=" * 80 + "\n\n")
        
        # Current Trends Section
        f.write("🚀 CURRENT TRENDS & EMERGING THEMES\n")
        f.write("-" * 50 + "\n")
        # Clean up any duplicated text and format nicely
        trends_text = research_summary.trends_info
        if "trends_info='" in trends_text:
            trends_text = trends_text.split("trends_info='")[-1].rstrip("'")
        f.write(trends_text + "\n\n")
        
        # Methodologies Section
        f.write("🔧 KEY METHODOLOGIES & APPROACHES\n")
        f.write("-" * 50 + "\n")
        methods_text = research_summary.methods
        if "methods='" in methods_text:
            methods_text = methods_text.split("methods='")[-1].rstrip("'")
        f.write(methods_text + "\n\n")
        
        # Limitations Section
        f.write("⚠️ LIMITATIONS & CHALLENGES\n")
        f.write("-" * 50 + "\n")
        limitations_text = research_summary.limitations
        if "limitations='" in limitations_text:
            limitations_text = limitations_text.split("limitations='")[-1].rstrip("'")
        f.write(limitations_text + "\n\n")
        
        # Footer
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
        
    print(f"✅ Research analysis successfully written to {filename}")


def write(state: AgentState) -> AgentState:
    if 'research_summary' in state:
        write_beautiful_answer_to_file(state['result_summary'].topic, state['research_summary'])
        state['file_written'] = True
        state['output_filename'] = "answer.txt"
    return state

builder = StateGraph(AgentState)
builder.add_node("ReActAgentNode", agent_node)
builder.add_node("SummaryNode", get_summary)
builder.add_node("WritterNode", write)

builder.add_edge(START, "ReActAgentNode")
builder.add_edge("ReActAgentNode", "SummaryNode")
builder.add_edge("SummaryNode", "WritterNode")
builder.add_edge("WritterNode", END)

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
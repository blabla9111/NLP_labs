from typing import List, Any, Literal, Dict
from pydantic import BaseModel, Field

from langchain_deepseek.chat_models import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from langgraph.graph import StateGraph, START, END

import arxiv
import os

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))


BASE_URL = os.getenv("LITELLM_BASE_URL", "http://a6k2.dgx:34000/v1")
API_KEY = os.getenv("LITELLM_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3-32b")

class ProblemSpec(BaseModel):
    """Problem specification for PINN"""
    problem_type: Literal["fluid_dynamics", "heat_transfer", "structural_mechanics", "epidemiology", "other"] = Field(description="Type of physical problem")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in classification from 0 to 1")
    reasoning: str = Field(description="Explanation for the chosen problem type")

class ResearchPaper(BaseModel):
    title: str = Field(default="Untitled Paper", description="Title of research paper")
    authors: str = Field(default="Unknown Authors", description="Comma-separated author names")
    abstract: str = Field(default="", description="Abstract content")
    url: str = Field(default="#", description="ArXiv URL")
    published: str = Field(default="Unknown date", description="Publication date")

def fetch_arxiv_papers(query: str, max_results = 3) ->  List[ResearchPaper]:
    client = arxiv.Client()
    search = arxiv.Search( query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)

    papers = []

    for result in client.results(search):
        authors = ", ".join(author.name for author in result.authors)
        papers.append(ResearchPaper(title=result.title,
                                    authors=authors,
                                    abstract=result.summary,
                                    url = result.entry_id,
                                    published=str(result.published.date())))
        
    return papers

parser = PydanticOutputParser(pydantic_object=ProblemSpec)

llm = ChatDeepSeek(
    api_base=BASE_URL,
    base_url=BASE_URL,
    api_key=API_KEY,
    model=MODEL_NAME,
    streaming=False,
    timeout=120
)

def send_to_arxiv(state: Dict[str, Any]) -> Dict[str, Any]:
    problem_type = state['problem_type']
    query = """Physics-Informed Neural Networks (PINN) in {problem_type} """# 
    print(fetch_arxiv_papers(query))

def get_problem_type(state: Dict[str, Any]) -> Dict[str, Any]:
    prompt = ChatPromptTemplate(messages=[
            ("system", """You are an expert in Physics-Informed Neural Networks (PINN). 
        Analyze the problem description and determine its type.
        {format_instructions}
        result_output = {schema}

        Return ONLY valid JSON in this exact format."""),
            
            ("human", "PROBLEM DESCRIPTION: {query}")
        ],
        partial_variables = {"format_instructions": parser.get_format_instructions()})

    chain = prompt | llm | parser

    out: ProblemSpec =  chain.invoke({"query":state['user_problem'], "schema": ProblemSpec.model_json_schema()})

    # 
    state['problem_type'] = out.problem_type
    send_to_arxiv(state)
    return state

def ask_approval(state:Dict[str, Any]) -> Dict[str, Any]:
    print("Your task is related to ")
    print(state["problem_type"])

    ans = input("Is it correct? (yes/no)\n> ")
    
    return {"approval": ans.strip().lower()}

class State(Dict[str, Any]):
    user_problem: str
    problem_type: str
    approval: str


builder = StateGraph(State)
builder.add_node("get_problem_type", get_problem_type)
builder.add_node("approval_node", ask_approval)

builder.add_edge(START, "get_problem_type")
builder.add_edge("get_problem_type", "approval_node")

def route(state: Dict[str, Any]) -> str:
    return END if state.get('approval', '').startswith('y') else 'get_problem_type'

builder.add_conditional_edges("approval_node", route, {"get_problem_type": "get_problem_type", END: END})

graph = builder.compile()

graph_png = graph.get_graph(xray=True)
png_bytes = graph_png.draw_mermaid_png()

with open("simple_agent.png", "wb") as f:
    f.write(png_bytes)

if __name__ == "__main__":
    final_state = graph.invoke({"user_problem":"We need to solve a problem of heat conduction in a rod"})
    print("Final user problem type is")
    print(final_state['problem_type'])
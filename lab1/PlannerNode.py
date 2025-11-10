from typing import Any, Dict, List
from pydantic import BaseModel, Field
from langchain_deepseek.chat_models import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from langgraph.graph import StateGraph, START, END

import arxiv
import os

from dotenv import load_dotenv, find_dotenv
import requests
load_dotenv(find_dotenv(usecwd=True))


BASE_URL = os.getenv("LITELLM_BASE_URL", "http://a6k2.dgx:34000/v1")
API_KEY = os.getenv("LITELLM_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3-32b")
GITHUB_API_TOKEN = os.getenv("GITHUB_API_TOKEN","")


class TopicSpec(BaseModel):
    topic: str =  Field(default="Unknown topic", description="Main topic of the user query")

class ResearchPaper(BaseModel):
    title: str = Field(default="Untitled Paper", description="Title of research paper")
    authors: str = Field(default="Unknown Authors", description="Comma-separated author names")
    abstract: str = Field(default="", description="Abstract content")
    url: str = Field(default="#", description="ArXiv URL")
    published: str = Field(default="Unknown date", description="Publication date")

class ResearchSummary(BaseModel):
    trends_info: str =  Field(default="No trends", description="Trends in the topic area")
    methods: str = Field(default="No methods", description="Used methods in the papers")
    limitations: str = Field(default="No limitations", description="Limitations and disadvantages in the papers")

class GithubReposInfo(BaseModel):
    name: str = Field(default="No Name", description="Name of github repository")
    description: str = Field(default="None", description="Description of the project in github")
    url: str = Field(default="No url", description="URL")




def get_topics(state: Dict[str, Any]) -> Dict[str, Any]:
    parser = PydanticOutputParser(pydantic_object=TopicSpec)
    prompt = ChatPromptTemplate(messages=[
        ("system", """You are an expert task analyzer. Extract and categorize tasks from user queries.
            Analyze the user query to identify main topic.
            {format_instructions}
            result_output = {schema}
            
            Return ONLY valid JSON in this exact format."""),
            
            ("human", "User query: {query}")
        ],
        partial_variables={"format_instructions": parser.get_format_instructions()})
    
    llm = ChatDeepSeek(
                        api_base=BASE_URL,
                        base_url=BASE_URL,
                        api_key=API_KEY,
                        model=MODEL_NAME,
                        streaming=False,
                        timeout=120
                        )
    
    chain = prompt | llm | parser

    output: TopicSpec = chain.invoke({"schema":TopicSpec.model_json_schema(), "query":state["query"]})
    # print(output)
    # state['topic'] = output.topic
    # arxiv_search(state['topic'])
    return {'topic': output.topic}

def arxiv_search(query, max_results = 3) -> List[ResearchPaper]:
    client = arxiv.Client()
    search = arxiv.Search(query= query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    papers = []

    for result in client.results(search):
        authors = ", ".join(author.name for author in result.authors)
        papers.append(ResearchPaper(title=result.title,
                                    authors=authors,
                                    abstract=result.summary,
                                    url = result.entry_id,
                                    published=str(result.published.date())))
        
    return papers

def get_summary(state:Dict[str, Any]) -> Dict[str, Any]:
    papers = arxiv_search(state['topic'], max_results=3)
    papers_info = "\n\n".join([
            f"Paper {i+1}: {p.title}\n"
            f"URL: {p.url}\n"
            f"Abstract: {p.abstract}\n"
            f"Published: {p.published}"
            for i, p in enumerate(papers)
        ])
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

        Research Papers Context:
        {query}

        Focus on extracting:
        1. TRENDS: Current research directions, popular approaches, and evolving themes in this field
        2. METHODS: Specific methodologies, algorithms, frameworks, and experimental approaches used
        3. LIMITATIONS: Explicitly stated limitations, methodological constraints, and areas needing improvement

        Return only the valid JSON object without any additional text.""")
        ],
        partial_variables={"format_instructions": parser.get_format_instructions()})
    
    llm = ChatDeepSeek(
                        api_base=BASE_URL,
                        base_url=BASE_URL,
                        api_key=API_KEY,
                        model=MODEL_NAME,
                        streaming=False,
                        timeout=120
                        )
    
    chain = prompt | llm | parser

    output: ResearchSummary = chain.invoke({"schema":ResearchSummary.model_json_schema(), "query":papers_info})
    print(output)
    # print()
    return {}

def search_github_repositories(search_query: str,github_token: str,max_results: int = 3) -> List[GithubReposInfo]:
    """
    Ищет репозитории на GitHub по ключевым словам
    """
    
    # Настраиваем заголовки с токеном
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'Authorization': f'token {github_token}'
    }
    
    # Выполняем поиск репозиториев
    url = "https://api.github.com/search/repositories"
    params = {
        'q': search_query,
        'per_page': min(max_results, 100),
        'sort': 'stars',  # Сортировка по звездам (самые популярные)
        'order': 'desc'
    }
    
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    
    search_data = response.json()
    results = []
    
    # Обрабатываем результаты
    for item in search_data.get('items', []):
        results.append(GithubReposInfo(name=item['full_name'], description=item.get('description', ''), url=item['html_url']))
    
    return results

def get_github_repos(state: Dict[str, Any]) -> Dict[str, any]:

    repos = search_github_repositories(state['topic'], GITHUB_API_TOKEN)
    print("\n\n\n!!!!!!!\n\n")
    print(state['topic'])
    for repo in repos:
        print(repo.url)

    return {}

class State(Dict[str, Any]):
    query: str
    approved: bool
    topic: str

builder = StateGraph(State)

builder.add_node("get_topics", get_topics)
builder.add_node("get_summary", get_summary)
builder.add_node("get_github_repos", get_github_repos)

builder.add_edge(START, "get_topics")
builder.add_edge("get_topics","get_summary")
builder.add_edge("get_topics","get_github_repos")

builder.add_edge("get_summary", END)
builder.add_edge("get_github_repos", END)

graph = builder.compile()
graph_png = graph.get_graph(xray=True)
png_bytes = graph_png.draw_mermaid_png()

with open("simple_agent.png", "wb") as f:
    f.write(png_bytes)

if __name__ == "__main__":
    final_state = graph.invoke({"query":"find please for me PINN in epidemiology"})
    print("Final state is")
    print(final_state['query'])
    print(final_state['topic'])



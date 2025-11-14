from pydantic import BaseModel, Field
from typing import List, TypedDict

class ResearchPaper(BaseModel):
    title: str = Field(default="Untitled Paper", description="Title of research paper")
    authors: str = Field(default="Unknown Authors", description="Comma-separated author names")
    abstract: str = Field(default="", description="Abstract content")
    url: str = Field(default="#", description="ArXiv URL")
    published: str = Field(default="Unknown date", description="Publication date")

class TopicSpec(BaseModel):
    topic: str =  Field(default="Unknown topic", description="Main topic of the user query")

class ResultSummary(BaseModel):
    topic: str = Field(default="Unknown topic", description="User query's main topic")
    # response: str = Field(default="None", description="Response for the user's query")
    arxiv_api_response: str = Field(default="None", description="Response for the user's query from ArXix API")
    crossref_api_response: str = Field(default="None", description="Response for the user's query from CrossRef API")

class ResearchSummary(BaseModel):
    trends_info: str =  Field(default="No trends", description="Trends in the topic area")
    methods: str = Field(default="No methods", description="Used methods in the papers")
    limitations: str = Field(default="No limitations", description="Limitations and disadvantages in the papers")

class GraphState(TypedDict):
    messages: List  # Сообщения (SystemMessage, HumanMessage)
    # intermediate_steps: Annotated[List[tuple], operator.add]  # Шаги выполнения
    current_response: str  # Текущий ответ
    result_summary: ResultSummary
    research_summary: ResearchSummary
from pydantic import BaseModel, Field

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
    response: str = Field(default="None", description="Response for the user's query")
from typing import Dict, Any, List

import arxiv
from pydantic import BaseModel, Field

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

# def analyze_papers(papers: List[ResearchPaper], query: str) -> str:
#     try:
#         papers_info = "\n\n".join([
#             f"Paper {i+1}: {p.title}\n"
#             f"URL: {p.url}\n"
#             f"Abstract: {p.abstract}\n"
#             f"Published: {p.published}"
#             for i, p in enumerate(papers)
#         ])

#         PINN_loss_function_prompt = 
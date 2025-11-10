
from input_output_formats import ResearchPaper, TopicSpec
from langchain.tools import tool

import arxiv



@tool(description="tool for find papers in the ArXiv website",args_schema=TopicSpec)
def arxiv_search(topic, max_results = 3) -> str:
    client = arxiv.Client()
    search = arxiv.Search(query= topic, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    papers = []

    for result in client.results(search):
        authors = ", ".join(author.name for author in result.authors)
        papers.append(ResearchPaper(title=result.title,
                                    authors=authors,
                                    abstract=result.summary,
                                    url = result.entry_id,
                                    published=str(result.published.date())))
        
    return str(papers)

from lab1.data_formats.input_output_formats import ResearchPaper

import arxiv
import random


def arxiv_search(topic, max_results=3) -> str:
    # print("ARXIV RESEARCH START\n\n")
    # Генерируем случайное число от 0 до 1
    random_value = random.random()

    # 40% вероятность ошибки
    if random_value < 0.2:
        errors = [
            ConnectionError("Timeout connecting to arXiv API"),
            TimeoutError("Server response timeout"),
            Exception("arXiv API is temporarily unavailable"),
            ConnectionError("Network connection lost"),
            Exception("Unexpected API error - please try again")
        ]
        raise random.choice(errors)
    client = arxiv.Client()
    search = arxiv.Search(query=topic, max_results=max_results,
                          sort_by=arxiv.SortCriterion.Relevance)
    papers = []

    for result in client.results(search):
        authors = ", ".join(author.name for author in result.authors)
        papers.append(ResearchPaper(title=result.title,
                                    authors=authors,
                                    abstract=result.summary,
                                    url=result.entry_id,
                                    published=str(result.published.date())))

    return str(papers)

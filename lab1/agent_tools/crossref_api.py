import requests
from typing import List, Dict
from lab1.input_output_formats import ResearchPaper
import random


def get_papers_from_crossref(query: str, rows: int = 3) -> List[Dict]:
    """
    Поиск статей в Crossref и извлечение структурированной информации
    """
    url = "https://api.crossref.org/works"
    params = {
        'query': query,
        'rows': rows,
        'select': 'DOI,title,abstract,author,created,URL'
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    papers = []
    
    for item in data['message']['items']:
        # Извлекаем заголовок
        title = "Untitled Paper"
        if 'title' in item and item['title']:
            title = item['title'][0]
        
        # Извлекаем авторов
        authors = "Unknown Authors"
        if 'author' in item and item['author']:
            author_names = []
            for author in item['author']:
                given = author.get('given', '')
                family = author.get('family', '')
                if given or family:
                    author_names.append(f"{given} {family}".strip())
            if author_names:
                authors = ", ".join(author_names)
        
        # Извлекаем абстракт
        abstract = ""
        if 'abstract' in item:
            # Очищаем от JATS тегов
            import re
            abstract = re.sub(r'<jats:p>|</jats:p>', '', item['abstract'])
            abstract = re.sub(r'<[^>]+>', '', abstract)
        
        # Извлекаем URL
        url = "#"
        if 'URL' in item:
            url = item['URL']
        elif 'DOI' in item:
            url = f"https://doi.org/{item['DOI']}"
        
        # Извлекаем дату публикации
        published = "Unknown date"
        if 'created' in item and 'date-parts' in item['created']:
            date_parts = item['created']['date-parts'][0]
            if len(date_parts) >= 3:
                published = f"{date_parts[0]}-{date_parts[1]:02d}-{date_parts[2]:02d}"
            elif len(date_parts) >= 1:
                published = str(date_parts[0])
        
        paper_data = {
            'title': title,
            'authors': authors,
            'abstract': abstract,
            'url': url,
            'published': published
        }
        
        papers.append(paper_data)
    
    return papers

def search_crossref(topic, max_results = 3) -> str:
    """
    Получает статьи из Crossref и возвращает как список ResearchPaper объектов
    """
    print("Crossref RESEARCH START\n\n")
    # Генерируем случайное число от 0 до 1
    random_value = random.random()
    
    # 40% вероятность ошибки
    if random_value < 0.4:
        errors = [
            ConnectionError("Timeout connecting to Crossref API"),
            TimeoutError("Server response timeout"),
            Exception("Crossref API is temporarily unavailable"),
            ConnectionError("Network connection lost"),
            Exception("Unexpected API error - please try again")
        ]
        raise random.choice(errors)
    papers_data = get_papers_from_crossref(topic, max_results)
    research_papers = []
    
    for paper_dict in papers_data:
        research_paper = ResearchPaper(**paper_dict)
        research_papers.append(research_paper)

    # print("\n\n")
    # print("research_papers_crossref_api_results")
    # print(research_papers)
    
    return research_papers
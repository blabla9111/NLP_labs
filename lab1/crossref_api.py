import requests
from typing import List, Dict, Optional
from input_output_formats import ResearchPaper


def search_crossref(query: str, rows: int = 5) -> List[Dict]:
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

# Пример использования
def get_structured_crossref_papers(query: str) -> List[ResearchPaper]:
    """
    Получает статьи из Crossref и возвращает как список ResearchPaper объектов
    """
    papers_data = search_crossref(query)
    research_papers = []
    
    for paper_dict in papers_data:
        research_paper = ResearchPaper(**paper_dict)
        research_papers.append(research_paper)
    
    return research_papers

# Использование
if __name__ == "__main__":
    papers = get_structured_crossref_papers("neural network")
    
    for i, paper in enumerate(papers, 1):
        print(f"{i}. {paper.title}")
        print(f"   Authors: {paper.authors}")
        print(f"   Published: {paper.published}")
        print(f"   URL: {paper.url}")
        print(f"   Abstract: {paper.abstract[:200]}...")
        print("-" * 60)
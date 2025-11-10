from typing import Dict, Any, List
import wikipediaapi
import requests
from pydantic import BaseModel, Field

class WikipediaArticle(BaseModel):
    title: str = Field(default="Untitled Article", description="Title of Wikipedia article")
    summary: str = Field(default="", description="Article summary")
    url: str = Field(default="#", description="Wikipedia URL")
    categories: List[str] = Field(default_factory=list, description="Article categories")
    sections: List[str] = Field(default_factory=list, description="Main sections")
    equations: List[str] = Field(default_factory=list, description="Found equations")


def fetch_wikipedia_articles(query: str, max_results: int = 3, language: str = 'en') -> List[WikipediaArticle]:
    """
    Search for Wikipedia articles related to the query
    
    Args:
        query: Search query
        max_results: Maximum number of articles to return
        language: Wikipedia language code ('en', 'ru', etc.)
    """
    
    # Initialize Wikipedia API
    wiki = wikipediaapi.Wikipedia(
        language=language,
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent="PINNResearchBot/1.0"
    )
    
    articles = []
    
    try:
        # Search for pages
        search_results = wiki.search(query, results=max_results)
        
        for page_title in search_results[:max_results]:
            page = wiki.page(page_title)
            
            if page.exists():
                # Extract equations from content
                equations = _extract_equations(page.text)
                
                # Get main sections
                sections = list(page.sections.keys())
                
                article = WikipediaArticle(
                    title=page.title,
                    summary=page.summary,
                    url=page.fullurl,
                    categories=list(page.categories.keys()),
                    sections=sections[:10],  # Limit to first 10 sections
                    equations=equations
                )
                articles.append(article)
                
    except Exception as e:
        print(f"Error fetching Wikipedia articles: {e}")
    
    return articles
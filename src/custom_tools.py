from ddgs import DDGS
from pydantic import BaseModel
from langchain.tools import tool

class SearchResults(BaseModel):
    title: str
    href: str
    body: str

@tool
def search_internet(query: str, max_results: int = 5):
    """
    Searches the internet using DuckDuckGo.
    
    Args:
        query (str): The search query.
        max_results (int): The maximum number of results to return. Defaults to 5.
        
    Returns:
        list: A list of dictionaries containing the search results (title, href, body).
    """
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results, backend="google, bing")
    pdResult = [SearchResults(**result) for result in results]
    return pdResult


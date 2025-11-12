from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.tools import tool

from lab1.agent_tools.arxiv_search_tool import arxiv_search
from lab1.agent_tools.crossref_api import search_crossref
from lab1.input_output_formats import TopicSpec

@tool(description="tool for find papers in the ArXiv and the CrossRef websites", args_schema=TopicSpec)
def do_research(topic) -> str:
    
    runnable = RunnableParallel(
        topic = RunnablePassthrough(),
        arxiv_api_response=lambda x: arxiv_search(x["topic"]), 
        crossref_api_response=lambda x: search_crossref(x["topic"])
    )
    
    result = runnable.invoke({"topic": topic})
    # print(str(result))
    return result



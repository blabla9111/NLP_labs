from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.tools import tool

from lab1.agent_tools.arxiv_search_tool import arxiv_search
from lab1.agent_tools.crossref_api import search_crossref
from lab1.input_output_formats import TopicSpec
from lab1.utils.retry_decorator import retry_on_failure


@tool(description="tool for find papers in the ArXiv and the CrossRef websites", args_schema=TopicSpec)
def do_research(topic) -> str:

    @retry_on_failure(max_retries=3, delay=1)
    def arxiv_with_retry(topic):
        return arxiv_search(topic)

    @retry_on_failure(max_retries=3, delay=1)
    def crossref_with_retry(topic):
        return search_crossref(topic)

    runnable = RunnableParallel(
        topic=RunnablePassthrough(),
        arxiv_api_response=lambda x: arxiv_with_retry(x["topic"]),
        crossref_api_response=lambda x: crossref_with_retry(x["topic"])
    )

    result = runnable.invoke({"topic": topic})
    return result

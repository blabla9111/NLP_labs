from langchain_deepseek.chat_models import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.tools import tool

from lab1.input_output_formats import TopicSpec
from lab1 import config



@tool(description="tool for find main topic from user query")
def get_topic(user_query) -> str:
    parser = PydanticOutputParser(pydantic_object=TopicSpec)
    prompt = ChatPromptTemplate(messages=[
        ("system", """You are an expert task analyzer. Extract and categorize tasks from user queries.
            Analyze the user query to identify main topic.
            {format_instructions}
            result_output = {schema}
            
            Return ONLY valid JSON in this exact format."""),
            
            ("human", "User query: {query}")
        ],
        partial_variables={"format_instructions": parser.get_format_instructions()})
    
    llm = ChatDeepSeek(
                        api_base=config.BASE_URL,
                        base_url=config.BASE_URL,
                        api_key=config.API_KEY,
                        model=config.MODEL_NAME,
                        streaming=False,
                        timeout=120
                        )
    
    chain = prompt | llm | parser

    output: TopicSpec = chain.invoke({"schema":TopicSpec.model_json_schema(), "query":user_query})
    return output.topic
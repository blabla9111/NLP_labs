from langchain_deepseek.chat_models import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.tools import tool

from lab1.data_formats.input_output_formats import TopicSpec
from lab1 import config


@tool(description="tool for find main topic from user query")
def get_topic(user_query) -> str:
    parser = PydanticOutputParser(pydantic_object=TopicSpec)
    prompt = ChatPromptTemplate(messages=[
                    ("system", """You are an expert scientific topic analyzer. Your task is to extract ONLY the main scientific topic from the user query.

                INSTRUCTIONS:
                1. Analyze the user query and identify the core scientific topic/research subject
                2. If a clear scientific topic is present, extract it concisely
                3. If no scientific topic can be identified, return "Unknown topic"
                4. Do NOT add any explanations, commentary, or additional text
                5. Return ONLY the topic name or "Unknown topic"

                EXAMPLES:
                - Input: "What are the latest developments in quantum computing?"
                Output: "quantum computing"

                - Input: "Find research papers about neural networks"
                Output: "neural networks"

                - Input: "Hello, how are you today?"
                Output: "Unknown topic"

                - Input: "Tell me about machine learning applications"
                Output: "machine learning"

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
        timeout=240
    )

    chain = prompt | llm | parser

    output: TopicSpec = chain.invoke(
        {"schema": TopicSpec.model_json_schema(), "query": user_query})
    return output.topic

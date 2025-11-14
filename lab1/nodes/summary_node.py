from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from lab1.input_output_formats import GraphState, ResearchSummary


class SummaryGenerator:
    def __init__(self, model, parser_output_class):
        self.llm = model
        self.parser = PydanticOutputParser(pydantic_object=parser_output_class)
        self.prompt = self._create_prompt_template()

    def _create_prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate(messages=[
            ("system", """You are an expert research assistant specialized in analyzing and summarizing academic papers. Your task is to extract key information from research papers and return structured summaries.

        {format_instructions}

        Analyze the provided research papers and return a JSON object with the following structure:
        - trends_info: Identify major trends, research directions, and emerging themes in the topic area
        - methods: Extract and categorize the research methodologies, algorithms, and techniques used across the papers  
        - limitations: Document the limitations, disadvantages, and gaps mentioned in the papers

        Return ONLY valid JSON in this exact format. Do not include any additional text, explanations, or markdown formatting."""),

            ("human", """Please analyze the following research papers and provide a comprehensive summary:

        Research topic:
             {topic}
                  
        Research Papers Context:
        ArXiv Papers
             {arxiv_papers_info}
        
        CrossRef Papers
             {crossref_papers_info}

        Focus on extracting:
        1. TRENDS: Current research directions, popular approaches, and evolving themes in this field
        2. METHODS: Specific methodologies, algorithms, frameworks, and experimental approaches used
        3. LIMITATIONS: Explicitly stated limitations, methodological constraints, and areas needing improvement

        Return only the valid JSON object without any additional text.""")
        ],
            partial_variables={"format_instructions": self.parser.get_format_instructions()})

    def generate_summary(self, state: GraphState) -> GraphState:
        print("START SUMMARIZING")

        topic = state["result_summary"].topic
        arxiv_papers_info = state["result_summary"].arxiv_api_response
        crossref_papers_info = state["result_summary"].crossref_api_response

        chain = self.prompt | self.llm | self.parser

        output: ResearchSummary = chain.invoke({
            "topic": topic,
            "arxiv_papers_info": arxiv_papers_info,
            "crossref_papers_info": crossref_papers_info
        })

        return {"research_summary": output}

    def __call__(self, state: GraphState) -> GraphState:
        return self.generate_summary(state)

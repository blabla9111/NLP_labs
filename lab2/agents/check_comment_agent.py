from langchain_core.output_parsers import PydanticOutputParser
from langchain_deepseek.chat_models import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.tools import tool

from lab2 import config
from lab2.utils.retry_parser import RetryParser
from lab2.data_formats.input_output_formats import ExpertComment, GraphState
from lab2.agents.retriever_agent import RetrieverAgent


class CheckCommentAgent:
    def __init__(self, model, parser_output_class):
        self.llm = model
        self.parser = PydanticOutputParser(pydantic_object=parser_output_class)
        self.prompt = self._create_prompt_template()

    def _create_prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate(messages=[
            ("system", """You are an expert validation assistant specialized in evaluating expert comments on epidemiological forecasts for quality and compliance. Your task is to analyze comments against specific validation criteria and return structured assessments.

                {format_instructions}

                Return ONLY valid JSON in this exact format. Do not include any additional text, explanations, or markdown formatting."""),

            ("human", """Please evaluate the following expert comment for compliance with validation criteria.

                **Validation Criteria for a Good Comment:**

                1.  **Specific & Useful:** The comment must describe a **clear, single observation or problem** in the forecast (e.g., "the peak is offset by 10 days," "the decline after the peak is too sharp," "the spread rate is underestimated"). General phrases (e.g., "the forecast is incorrect," "need to be recalculated," "the graph doesn't match reality") are unacceptable.
                2.  **Clear Purpose:** The comment's purpose must be clearly stated—to clarify, correct, adjust, or change a **specific aspect** of the forecast.
                3.  **Focused:** The comment should address **only one key issue**.
                4.  **Objective Tone:** The comment must be free of emotion, subjective judgments, expressions of dissatisfaction, and unsubstantiated assumptions.

                **Expert Comment to Validate:**
                {expert_comment}

                **Similar Expert Comments from Database:**
                {similar_comments}

                **Additional Instructions:**
                - Compare the new comment with the similar comments from the database
                - Use the database examples as reference for what constitutes a valid comment
                - If the new comment is similar to valid comments in the database, it's more likely to be valid
                - Note that database comments may vary in quality - use them as examples, not absolute standards

                Return only the valid JSON object without any additional text.""")
        ],
            partial_variables={"format_instructions": self.parser.get_format_instructions()})

    def check_expert_comment(self, state: GraphState) -> GraphState:
        print("Check expert comment")
        expert_comment = state["expert_comment"].comment
        print(expert_comment)
        retriever = RetrieverAgent()
        similar_comments = retriever.similarity_search(text = expert_comment)

        chain = self.prompt | RunnableParallel(output=self.llm, prompt=RunnablePassthrough(
        )) | RetryParser(llm=self.llm, parser=self.parser)

        output: ExpertComment = chain.invoke({
            "expert_comment": expert_comment,
            "similar_comments": str(similar_comments)
        })

        return {"expert_comment": output}
    
    def check_expert_comment_tool(self, expert_comment: str) -> str:

        retriever = RetrieverAgent()
        similar_comments = retriever.similarity_search(text = expert_comment)

        chain = self.prompt | RunnableParallel(output=self.llm, prompt=RunnablePassthrough(
        )) | RetryParser(llm=self.llm, parser=self.parser)

        output: ExpertComment = chain.invoke({
            "expert_comment": expert_comment,
            "similar_comments": similar_comments
        })

        return str(output)

    def __call__(self, state: GraphState) -> GraphState:
        return self.check_expert_comment(state)
    

@tool(description="tool to check expert_comment validity")
def check_comment_validity(expert_comment) -> str:
    # parser = PydanticOutputParser(pydantic_object=ExpertComment)
    llm = ChatDeepSeek(
        api_base=config.BASE_URL,
        base_url=config.BASE_URL,
        api_key=config.API_KEY,
        model=config.MODEL_NAME,
        streaming=False,
        timeout=240
    )
    agent = CheckCommentAgent(model=llm, parser_output_class=ExpertComment)
    output = agent.check_expert_comment_tool(expert_comment=expert_comment)
    print(output)
    return output

@tool(description="Asks the user to enter a new comment if the previous one was not valid (is_valid = False). After that check comment validity again")
def get_new_comment_from_expert(reason, recommendations) -> str:
    user_query = input(
        f"Your previous comment can't be used, because {reason}\n. There are some recommendations {recommendations}. \nWrite, please, new one: ")
    return user_query.strip()
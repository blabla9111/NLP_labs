from langchain.tools import tool
from langchain_deepseek.chat_models import ChatDeepSeek
from lab2.data_formats.input_output_formats import ExpertComment
from lab2 import config
from lab2.agents.check_comment_agent import CheckCommentAgent


@tool(description="tool to check expert_comment validity")
def check_comment_validity(expert_comment) -> str:
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
    return output


@tool(description="Asks the user to enter a new comment if the previous one was not valid (is_valid = False). After that check comment validity again")
def get_new_comment_from_expert(reason, recommendations) -> str:
    user_query = input(
        f"Your previous comment can't be used, because {reason}\n. There are some recommendations {recommendations}. \nWrite, please, new one: ")
    return user_query.strip()

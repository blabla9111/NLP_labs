from pydantic import BaseModel, Field
from typing import List, TypedDict


class GraphState(TypedDict):
    messages: List  # Сообщения (SystemMessage, HumanMessage)
    current_response: str  # Текущий ответ
    expert_comment: str

class ExpertComment(BaseModel):
    comment: str = Field(default = "No comment", description="expert's comment")
    is_valid: bool = Field(default = "False", description="Expert's comment fit in Validation Criteria: 1. CONCRETENESS 2. CERTAINTY 3. OBJECTIVITY. If comment is valid return True else False")
    reason: str = Field(default = "None", description = "If comment is not valid, describe your decision")

class GithubReposInfo(BaseModel):
    name: str = Field(default="No Name",
                      description="Name of github repository")
    description: str = Field(
        default="None", description="Description of the project in github")
    url: str = Field(default="No url", description="URL")

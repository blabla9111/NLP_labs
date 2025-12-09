from pydantic import BaseModel, Field
from typing import List, TypedDict


class ExpertComment(BaseModel):
    comment: str = Field(default="No comment", description="expert's comment")
    is_valid: bool = Field(
        default="False", description="Expert's comment fit in Validation Criteria: 1. CONCRETENESS 2. CERTAINTY 3. OBJECTIVITY. If comment is valid return True else False")
    reason: str = Field(
        default="None", description="If comment is not valid, describe your decision")
    recommendations: str = Field(
        default="None", description="If comment is not valid, give recommendations to create good expert comment")
    comment_class: str = Field(
        default="None", description="Call get_class_subclass_names to get expert comment class")
    comment_subclass: str = Field(
        default="None", description="Call get_class_subclass_names to get expert comment subclass")


class PINNLossWeights(BaseModel):
    data_loss_weight: float = Field(
        default="100.0", description="Weight for MSE loss between predicted and observed SIRD values. Range: [0, 100]")
    ODE_loss_weight: float = Field(
        default="1.0", description="Weight for physics-informed loss enforcing SIRD ODE constraints. Range: [0, 10]")
    initial_boundary_conditions_loss_weight: float = Field(
        default="0.1", description="Weight for boundary condition loss (I(t_end) → 0). Range: [0, 1]")
    peak_height_loss_weight: float = Field(
        default="0.0", description="Weight for penalty on deviation from expected infection peak height. Range: [0, 1]")
    slow_growth_penalty_weight: float = Field(
        default="0.0", description="Weight for penalizing unrealistically slow infection growth rates before peak. Range: [0, 1]")
    rapid_growth_penalty_weight: float = Field(
        default="0.0", description="Weight for penalizing unrealistically rapid infection growth rates before peak. Range: [0, 1]")
    reason: str = Field(default="None", description="Describe your decision")


class WeightValidationResult(BaseModel):
    is_valid: bool = Field(description="Whether all weights are valid")
    errors: List[str] = Field(default_factory=list,
                              description="List of validation errors")
    suggested_fixes: List[str] = Field(
        default_factory=list, description="Suggested fixes for invalid weights")
    reason: str = Field(description="Explanation of validation results")


class GraphState(TypedDict):
    messages: List  # Сообщения (SystemMessage, HumanMessage)
    current_response: str  # Текущий ответ
    expert_comment: ExpertComment
    loss_weights: PINNLossWeights
    validation_errors: List[str]
    current_agent: str
    handoff_count: int  # Кол-во передач управления
    session_id: str

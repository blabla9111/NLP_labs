from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from lab2.data_formats.input_output_formats import GraphState
from lab2.utils.retry_parser import RetryParser


class WeightChecker:
    def __init__(self, model, parser_output_class):
        self.llm = model
        self.output_class = parser_output_class
        self.parser = PydanticOutputParser(pydantic_object=parser_output_class)
        self.prompt = self._create_prompt_template()

    def _create_prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are a PINN loss weight validator. Your task is to check if the generated loss weights are valid and within acceptable ranges.
            
            {format_instructions}
            
            Return ONLY valid JSON in this exact format. Do not include any additional text."""),
            
            ("human", """## Weight Validation Task
            
            Generated Weights to Validate:
            {weights}
            
            ## Validation Rules:
            1. **data_loss_weight**: Must be between 0 and 100 (inclusive)
            2. **ODE_loss_weight**: Must be between 0 and 10 (inclusive)
            3. All other weights must be between 0 and 1 (inclusive)
            4. **reason** field must be present and non-empty
            
            ## Expert Context:
            Expert Comment: {expert_comment}
            Comment Class: {comment_class}
            Comment Subclass: {comment_subclass}
            
            ## Your Validation:
            - Check each weight against its range
            - If weights are invalid, suggest appropriate values
            - Consider if the weights appropriately address the expert's concerns
            - Provide clear reasons for any validation failures
            
            Return the validation results as specified.""")
        ]).partial(format_instructions=self.parser.get_format_instructions())

    def validate_weights(self, state: GraphState) -> GraphState:
        weights = state["loss_weights"]
        expert_comment = state["expert_comment"]
        
        # First, do basic range checking
        validation_errors = []
        
        weights_dict = weights.dict()
        
        # Check data_loss_weight
        data_weight = weights_dict.get("data_loss_weight", 0)
        if not (0 <= data_weight <= 100):
            validation_errors.append(f"data_loss_weight ({data_weight}) must be between 0 and 100")
        
        # Check ODE_loss_weight
        ode_weight = weights_dict.get("ODE_loss_weight", 0)
        if not (0 <= ode_weight <= 10):
            validation_errors.append(f"ODE_loss_weight ({ode_weight}) must be between 0 and 10")
        
        # OPTION 2: Direct attribute access (cleaner)
        # Check data_loss_weight
        if not (0 <= weights.data_loss_weight <= 100):
            validation_errors.append(f"data_loss_weight ({weights.data_loss_weight}) must be between 0 and 100")
        
        # Check ODE_loss_weight
        if not (0 <= weights.ODE_loss_weight <= 10):
            validation_errors.append(f"ODE_loss_weight ({weights.ODE_loss_weight}) must be between 0 and 10")
        
        # Check other weights (0-1 range)
        other_weight_attrs = [
            ("initial_boundary_conditions_loss_weight", weights.initial_boundary_conditions_loss_weight),
            ("peak_height_loss_weight", weights.peak_height_loss_weight),
            ("slow_growth_penalty_weight", weights.slow_growth_penalty_weight),
            ("rapid_growth_penalty_weight", weights.rapid_growth_penalty_weight)
        ]
        
        for name, weight in other_weight_attrs:
            if not (0 <= weight <= 1):
                validation_errors.append(f"{name} ({weight}) must be between 0 and 1")
        
        # Check reason field
        if not hasattr(weights, 'reason') or not weights.reason or weights.reason.strip() == "":
            validation_errors.append("Reason field must be present and non-empty")
        
        # If basic validation passes, use LLM for semantic validation
        if not validation_errors:
            chain = self.prompt | self.llm | self.parser
            validation_result = chain.invoke({
                "weights": weights_dict,  # Pass as dict
                "expert_comment": state["expert_comment"].comment,
                "comment_class": state["expert_comment"].comment_class,
                "comment_subclass": state["expert_comment"].comment_subclass
            })
            
            if not validation_result.is_valid:
                validation_errors.extend(validation_result.errors)
        
        return {
            "validation_errors": validation_errors,
            "current_agent": "WeightChecker"
        }

    def __call__(self, state: GraphState) -> GraphState:
        return self.validate_weights(state)

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from lab2.data_formats.input_output_formats import GraphState, PINNLossWeights
from lab2.utils.retry_parser import RetryParser


class PINNLossWieghtsGenerator:
    def __init__(self, model, parser_output_class):
        self.llm = model
        self.parser = PydanticOutputParser(pydantic_object=parser_output_class)
        self.prompt = self._create_prompt_template()

    def _create_prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate(messages=[
            ("system", """
            You are a PINN (Physics-Informed Neural Network) loss weights adjustment expert.
            Your task is to analyze expert comments on PINN predictions and adjust the loss weights accordingly.
            
            {format_instructions}
            
            Return ONLY valid JSON in this exact format. Do not include any additional text, explanations, or markdown formatting."""),

            ("human", """
            ## Expert Analysis Context
            An initial PINN model for SIRD (Susceptible-Infected-Recovered-Deceased) epidemic forecasting was trained with these base weights:
            - data_loss_weight = 100.0
            - ODE_loss_weight = 1.0
            - initial_boundary_conditions_loss_weight = 0.1
            - peak_height_loss_weight = 0.0
            - slow_growth_penalty_weight = 0.0
            - rapid_growth_penalty_weight = 0.0
            
            The model predictions have been reviewed by an expert, and adjustments are needed.
            
            ## Expert Classification
            Comment Class: {comment_class}
            Comment Subclass: {comment_subclass}
            
            ## Expert Comment
            {comment}
            
            ## Your Task
            Based on the expert's comment and classification, adjust the loss weights to address the identified issues.
            Consider the following guidelines:
            
            1. **data_loss_weight [0, 100]**: 
            - Increase (up to 100) if the model needs to better fit observed data
            - Decrease if overfitting to noisy data is a concern
            
            2. **ODE_loss_weight [0, 10]**:
            - Increase (up to 10) if the physics/biology constraints are being violated
            - Decrease if the ODE constraints are too restrictive
            
            3. **initial_boundary_conditions_loss_weight [0, 1]**:
            - Increase if boundary conditions (I→0 at end) are not satisfied
            - Typical range: 0.01-0.5
            
            4. **peak_height_loss_weight [0, 1]**:
            - Increase if infection peak height is significantly off from expert expectation
            - Use sparingly (0.01-0.1) for mild corrections
            
            5. **slow_growth_penalty_weight [0, 1]**:
            - Increase if growth before peak is unrealistically slow
            - Typical range: 0.01-0.2
            
            6. **rapid_growth_penalty_weight [0, 1]**:
            - Increase if growth before peak is unrealistically rapid
            - Typical range: 0.01-0.2
            
            ## Weight Adjustment Strategy
            - Make focused adjustments: Only change weights relevant to the specific issue
            - Use conservative increments/decrements (e.g., 1.5x, 0.5x of current value)
            - Maintain balance: Don't set all weights to extreme values
            - For new penalties not in base config, start with small values (0.01-0.05)
             
            ## Reason Field Requirement
            In the 'reason' field, provide a concise explanation that:
            1. References the expert comment and classification
            2. Explains which specific issues you're addressing
            3. Justifies your weight adjustments with rationale
            4. Mentions any trade-offs you considered
            
            Return only the adjusted weights as valid JSON.""")
        ],
            partial_variables={"format_instructions": self.parser.get_format_instructions()})

    def generate_weights(self, state: GraphState) -> GraphState:
        print("PINNLossWeightsGenerator")
        print(state["expert_comment"])
        comment = state["expert_comment"].comment
        comment_class = state["expert_comment"].comment_class
        comment_subclass = state["expert_comment"].comment_subclass

        chain = self.prompt | RunnableParallel(output=self.llm, prompt=RunnablePassthrough(
        )) | RetryParser(llm=self.llm, parser=self.parser)

        output: PINNLossWeights = chain.invoke({
                                                "comment": comment,
                                                "comment_class": comment_class,
                                                "comment_subclass": comment_subclass
                                            })

        return {
                "messages": state["messages"] + [output],
                "loss_weights": output,
                "handoff_count": state["handoff_count"] +1,
                "current_agent":"PINNLossWeightsGenerator"}

    def __call__(self, state: GraphState) -> GraphState:
        return self.generate_weights(state)

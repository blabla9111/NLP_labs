from lab1.data_formats.input_output_formats import ResultSummary, GraphState
from langchain.agents.structured_output import ToolStrategy
from langchain.agents import create_agent


class AgentNode:
    def __init__(self, model, tools, response_format=ToolStrategy(ResultSummary)):
        self.agent = create_agent(
            model=model, tools=tools, response_format=response_format)

    def __call__(self, state: GraphState) -> GraphState:
        chunks = self.agent.stream({"messages": state["messages"]})

        response_parts = []
        structured_response = None

        for chunk in chunks:
            response_parts.append(str(chunk))

            if isinstance(chunk, dict) and chunk.get('model', {}).get('structured_response'):
                structured_response = chunk['model']['structured_response']

        return {
            "current_response": structured_response,
            "messages": state["messages"] + [response_parts],
            "result_summary": structured_response
        }

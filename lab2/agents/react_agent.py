from langchain.agents.structured_output import ToolStrategy
from langchain.agents import create_agent

from lab2.data_formats.input_output_formats import GraphState, ExpertComment


class ReActAgent:
    def __init__(self, model, tools, response_format=ToolStrategy(ExpertComment)):
        self.agent = create_agent(
            model=model, tools=tools, response_format=response_format)

    def __call__(self, state: GraphState) -> GraphState:
        chunks = self.agent.stream({"messages": state["messages"]})

        response_parts = []
        structured_response = None

        for chunk in chunks:
            response_parts.append(str(chunk))

            if isinstance(chunk, dict) and 'model' in chunk and 'messages' in chunk['model']:
                for msg in chunk['model']['messages']:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            print(
                                f"The tool is called: {tool_call.get('name')}")

            if isinstance(chunk, dict) and chunk.get('model', {}).get('structured_response'):
                structured_response = chunk['model']['structured_response']

        return {
            "current_response": structured_response,
            "current_agent": "CleanerNode",
            "messages": state["messages"] + [response_parts],
            "expert_comment": structured_response
        }

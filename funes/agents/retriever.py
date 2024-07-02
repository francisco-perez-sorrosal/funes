import json

from typing import Annotated, Literal, Optional, TypedDict

from langchain_huggingface import ChatHuggingFace
from langgraph.graph import StateGraph, END

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages
from langgraph.pregel import GraphRecursionError
from langchain_core.messages import ToolMessage

from funes.agents.base_agent import BaseAgent
from funes.tools import ArxivSearchTool


class RetrieverAgentState(TypedDict):
    # The `add_messages` function within the annotation defines
    # *how* updates should be merged into the state.
    messages: Annotated[list, add_messages]
    # extracted_paper: str


# class Paper(BaseModel):
#     title: str = Field(description="the title of the paper")
#     arxiv_id: str = Field(description="the arxiv identifier of the paper")


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}
        print(f"Tools:\n{self.tools_by_name}")

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for i, tool_call in enumerate(message.tool_calls):
            print(f"Tool call {i}: {tool_call}")
            if tool_call["name"] not in self.tools_by_name:
                outputs.append(ToolMessage(
                    content=f"Tool {tool_call['name']} not found. Continuing...", 
                    name=tool_call["name"], 
                    tool_call_id=tool_call["id"]
                ))
                continue
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


class RetrieverAgent(BaseAgent):

    def __init__(self, lm, model_id: Optional[str] = "meta-llama/Meta-Llama-3-8B-Instruct", name: str = "retriever_agent"):
        super().__init__(name=name)
        print(f"Creating agent {name}")
        self.graph_definition = StateGraph(RetrieverAgentState)
        arxiv_search_tool = ArxivSearchTool()
        self.tools = [arxiv_search_tool]
        self.chat = ChatHuggingFace(llm=lm, model_id=model_id)
        self.chat = self.chat.bind_tools(self.tools)
        print(f"Chat model {self.chat.model_id} for agent {self.name} created")


    # Define the function that determines whether to continue or not
    def _tool_router(self, state: RetrieverAgentState) -> Literal["tools_route", "end"]:
        print(state)
        if isinstance(state, list):
            ai_message = state[-1]
        elif messages := state.get("messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools_route"
        return "end"
        
    def _create_graph(self):
        # Define the two nodes we will cycle between
        
        def chatbot(state: RetrieverAgentState):
            return {"messages": [self.chat.invoke(state["messages"])]}
        
        # Nodes
        self.graph_definition.add_node("chatbot", chatbot)
        self.graph_definition.add_node("tools", BasicToolNode(self.tools))

        # Edges
        self.graph_definition.add_edge("tools", "chatbot")
        # We now add a conditional edge
        self.graph_definition.add_conditional_edges(
            "chatbot",
            self._tool_router,
            {
                # If `tools`, then we call the tool node.
                "tools_route": "tools",
                "end": END
            }
        )
        
        # Entry point
        self.graph_definition.set_entry_point("chatbot")

        # Compile the graph
        self.graph = self.graph_definition.compile()

    def __call__(self, text: str):
        super().__call__()
        
        inputs = {"messages": [HumanMessage(content=text)]}
        config = RunnableConfig(recursion_limit=100)
        try:
            events = self.graph.stream(inputs, config, stream_mode="values")
        except GraphRecursionError:
            print("Graph recursion limit reached.")
            events = ["N/A"]
        return events

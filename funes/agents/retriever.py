
import arxiv
from typing import Annotated, Literal, Optional, Type

from typing_extensions import TypedDict

from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from langgraph.graph import StateGraph, END

from outlines import models

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


from langgraph.prebuilt import ToolInvocation
import json
from langchain_core.messages import FunctionMessage
from langgraph.prebuilt import ToolExecutor

from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.pregel import GraphRecursionError
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool

class RetrieverAgentState(TypedDict):
    # The `add_messages` function within the annotation defines
    # *how* updates should be merged into the state.
    messages: Annotated[list, add_messages]
#     # extracted_paper: str


# extracted_paper_examples = [
# { 
#      "example_input": "Get the Transformer paper", 
#      "model_output": """
#       {{
#         'title': 'Transformer',
#         'arxiv_id': ''
#       }}"""
# },
# { 
#      "example_input": "Retrieve the arxiv paper 1706.03762", 
#      "model_output": """
#       {{
#         'title': '',
#         'arxiv_id': '1706.03762'
#       }}"""
# },
# { 
#      "example_input": "Retrieve the Attention is all you need paper (arxiv 1706.03762)", 
#      "model_output": """
#       {{
#         'title': 'Attention is all you need',
#         'arxiv_id': '1706.03762'
#       }}"""
# },
# ]

# system_message_template = """You are an experienced text processing agent working with user documents that contain
# information about research papers. You have been asked to extract titles of papers and arxiv identifiers from user strings. 
# """

# prefix="""Extract specified values from the source text delimited by ### below. Return answer as JSON object with following fields:
# - 'title' <string>
# - 'arxiv_id' <string>

# Arxiv identifiers are strings that start with four numbers, then dot and finally five numbers, like 1706.03762.
# If you can't extract the or the arxiv identifier, add an empty string to the corresponding json field, e.g. {{'arxiv_id': ''}}. 
# Do not add any text for the title or the arxiv identifier that is not in the user string if you can't extract anything useful.

# Examples:"""

# example_prompt = PromptTemplate(
#     input_variables=["example_input", "model_output"], template="User input: {example_input}\n{model_output}"
# )

# user_prompt = FewShotPromptTemplate(
#     examples=extracted_paper_examples,
#     example_prompt=example_prompt,
#     prefix=prefix,
#     suffix="User input:\n### {user_input} ###",
#     input_variables=["user_input"],
# )


class SearchInput(BaseModel):
    query: str = Field(description="a search query")
    
class ArxivSearchTool(BaseTool):
    name = "arxiv_search"
    description = "useful for when you need to retrieve info about papers in arxiv"
    args_schema: Type[BaseModel] = SearchInput

    def _run(
        self, query: str, #run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the arxiv search tool."""
        
        client = arxiv.Client()
        search = arxiv.Search(
            query = query,
            max_results = 10,
            sort_by = arxiv.SortCriterion.SubmittedDate
        )

        arxiv_docs = client.results(search)
        
        arxiv_documents = []
        for doc in arxiv_docs:
            arxiv_documents.append({'title': doc.title})
        if len(arxiv_documents) == 0:
            return ""
        json_output = json.dumps({'docs': arxiv_documents})
        return json_output


    async def _arun(
        self, query: str, #run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


# class Paper(BaseModel):
#     title: str = Field(description="the title of the paper")
#     arxiv_id: str = Field(description="the arxiv identifier of the paper")
    

from langchain_core.messages import ToolMessage


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


class RetrieverAgent():

    def __init__(self, lm, model_id: Optional[str] = "meta-llama/Meta-Llama-3-8B-Instruct", name: str = "retriever_agent"):        
        self.workflow = StateGraph(RetrieverAgentState)
        self.app = None
        self.chat = ChatHuggingFace(llm=lm, model_id=model_id)
        arxiv_search_tool = ArxivSearchTool()
        tools = [arxiv_search_tool]
        self.chat = self.chat.bind_tools(tools)
        # self.tool_executor = ToolExecutor(tools)
        print(f"Chat model: {self.chat.model_id}")


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


#     # Define the function that extract the title or id
#     def _extract_title_or_id(self, state):
#         system_message = SystemMessage(system_message_template)
#         messages = state['messages']
        
#         print(f"Messages: {messages}")
        
#         human_message_text = user_prompt.format(user_input=str(messages[0].content))
#         print(human_message_text)
#         human_message = HumanMessage(human_message_text)
#         print(f"System message:\n{system_message}")
#         print(f"Human message:\n{human_message}")
#         # chain = system_message | human_message | parser
#         print(type(self.lm))
#         extracted_paper = self.lm.invoke([system_message, human_message], grammar={"type": "json", "value": Paper.schema()},)
        
        
#         print(f"Extracted paper: {extracted_paper}")
#         print(f"Type of extracted paper: {type(extracted_paper)}")
#         paper = Paper.parse_raw(extracted_paper)
#         print(f"Extracted paper 1: {paper}")
#         print(f"Type of extracted paper 1: {type(paper)}")
        
#         # We return a list, because this will get added to the existing list
#         state["extracted_paper"] = paper
    
#         state["messages"] = state["messages"] + [("paper_extracted")]
#         print(f"State: {state}")
#         return state
        



#     # # Define the function to execute tools
#     # def _call_retriever_tool(self, state):
#     #     extracted_paper = state["extracted_paper"]
        
#     #     print(f"Extracted paper in retriever tool: {extracted_paper}")
        
        
#     #     messages = state['messages']
#     #     last_message = messages[-1]
        
#     #     print(f"Last message: {last_message}")
        
#     #     action = ToolInvocation(
#     #         tool=last_message.additional_kwargs["function_call"]["name"],
#     #         tool_input=json.loads(last_message.additional_kwargs["function_call"]["arguments"]),
#     #     )
#     #     response = self.tool_executor.invoke(action)
#     #     function_message = FunctionMessage(content=str(response), name=action.tool)
        
#     #     return {"messages": [function_message]}        
        
    def _define_workflow(self):
        # Define the two nodes we will cycle between
        
        def chatbot(state: RetrieverAgentState):
            return {"messages": [self.chat.invoke(state["messages"])]}
        
        self.workflow.add_node("chatbot", chatbot)
        self.workflow.add_node("tools", BasicToolNode([ArxivSearchTool()]))

        self.workflow.set_entry_point("chatbot")
        self.workflow.add_edge("tools", "chatbot")

        # We now add a conditional edge
        self.workflow.add_conditional_edges(
            "chatbot",
            self._tool_router,
            {
                # If `tools`, then we call the tool node.
                "tools_route": "tools",
                "end": END
            }
        )

        # self.workflow.add_edge('retrieve_paper_action', 'title_extractor')
        self.app = self.workflow.compile()

    def __call__(self, text: str):        
        if not self.app:
            self._define_workflow()
            
        inputs = {"messages": [HumanMessage(content=text)]}
        config = RunnableConfig(recursion_limit=100)
        
        try:
            # result = self.app.invoke(inputs, config)
            events = self.app.stream(inputs, config, stream_mode="values")
        except GraphRecursionError:
            print("Graph recursion limit reached.")
            # result = ""
            events = ["N/A"]
        # return result
        return events
        
    def get_graph(self):
        if not self.app:
            self._define_workflow()
        return self.app.get_graph()

import functools
import operator

from langchain import hub

from typing import Annotated, List, Sequence, TypedDict, Optional, Union

from langchain.agents import AgentExecutor, create_openai_tools_agent, create_json_chat_agent
from langchain_core.tools import render_text_description
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.json_chat.prompt import TEMPLATE_TOOL_RESPONSE
from langchain.agents.format_scratchpad import format_log_to_messages
from langchain.agents.output_parsers import JSONAgentOutputParser

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableConfig, RunnablePassthrough
from langchain_core.runnables.utils import Input, Output
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_huggingface import ChatHuggingFace
from langchain.tools import BaseTool
from langchain.tools.render import render_text_description_and_args
from langgraph.graph import END, StateGraph, START
from langgraph.pregel import GraphRecursionError, StreamMode

from funes.tools import ArxivSearchTool, SmalltalkTool

   
class AgentCoordinator(Runnable):
    """A coordinator for agents (an LLM node). It just picks the next agent to process
    and decides when the work is completed"""
        
    members = ["PaperInfoRetriever"] #, "PaperDownloader"]
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        " following workers:  {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )
    supervisor_options = ["FINISH"] + members

    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}"
            " Avoid punctuation.",
        ),
    ]
    ).partial(options=str(supervisor_options), members=", ".join(members))
    
    
    route_tool_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": supervisor_options},
                    ],
                }
            },
            "required": ["next"],
        },
    }
    
    
    def __init__(self, lm, model_id: Optional[str] = "meta-llama/Meta-Llama-3-8B-Instruct") -> None:
        # kwargs = {"grammar":{"type": "json", "schema": AgentCoordinator.route_tool_def}}
        self.chat = ChatHuggingFace(llm=lm, model_id=model_id) #, **kwargs)
        self.chat = self.chat.bind_tools([AgentCoordinator.route_tool_def], tool_choice="route")
        
        
        self.chain = (
            AgentCoordinator.prompt | self.chat | JsonOutputToolsParser(tools=[AgentCoordinator.route_tool_def]) # JsonOutputFunctionsParser() we can't use this because it doesn't return the output
        )
        
    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        response = self.chain.invoke(input, config)
        print(f"Type Response: {type(response)}")
        print(f"Response: {response}")
        return response[-1]['args']
        
        # return {"messages": [response], "next": response.content}
        
    
    @staticmethod
    def create_agent(lm, model_id, tools: List[BaseTool], system_prompt: str, type: str = "json"):
        """Create an agent."""        
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system", system_prompt,
                ),
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        
        prompt_template = prompt_template.partial(
            tools=render_text_description(list(tools)),
            tool_names=", ".join([t.name for t in tools]),
        )
        
        # prompt = prompt_template.format_messages(
        #     messages=[HumanMessage(content="XXXXXXXXX", name="XXX")],
        #     agent_scratchpad=[AIMessage(content="YYYYYYYYY", name="YYY")],
        # )

        # print(f"Prompt: {prompt}")

        # prompt = hub.pull("hwchase17/react-chat-json")
        chat = ChatHuggingFace(llm=lm, model_id=model_id)
        chat = chat.bind_tools(tools=tools)
        chat = chat.bind(stop=["\nObservation"])

        agent = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: format_log_to_messages(
                    x["intermediate_steps"], template_tool_response=TEMPLATE_TOOL_RESPONSE
                )
            )
            | prompt_template
            | chat
            # | JSONAgentOutputParser() 
            # | JsonOutputToolsParser(tools=tools)
        )



        
        
        
        # # prompt = hub.pull("hwchase17/react-chat-json")
        # chat = ChatHuggingFace(llm=lm, model_id=model_id)
        # # chat = chat.bind_tools(tools=tools)
        # agent = create_json_chat_agent(chat, tools, prompt_template, stop_sequence=True)
        
        # llm_with_tools = chat.bind_tools(tools=tools) #chat.bind(tools=[convert_to_openai_tool(tool) for tool in tools])

        # agent = (
        #     # RunnablePassthrough.assign(
        #     #     agent_scratchpad=lambda x: format_to_openai_tool_messages(
        #     #         x["intermediate_steps"]
        #     #     )
        #     # )
        #     prompt
        #     | llm_with_tools
        #     # | JsonOutputToolsParser(tools=tools)
        # )

        return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

class AgentState(TypedDict):
    # The agent state is the input to each node in the graph
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str


class ArxivRetrieverAgenticGraph:
    """A graph that routes between agents."""
    def __init__(self, lm, model_id: Optional[str] = "meta-llama/Meta-Llama-3-8B-Instruct") -> None:


        def agent_node(state: AgentState, agent, name):
            print(f"State in agent {name} node: {state}")
            result = agent.invoke(state)
            print(type(result))
            print(f"Agent node result: {result}")
            return {"messages": [HumanMessage(content=result[-1]["args"], name=name)]}

        arxiv_search_tool = ArxivSearchTool()
        
        tools = [arxiv_search_tool, SmalltalkTool()]
        retriever_prompt_template = f"""You are an arxiv paper info researcher.
        You have access to the following tools:

        {render_text_description_and_args(tools).replace('{', '{{').replace('}', '}}')}
        
        The way you use the tools is by specifying a json blob.
        Specifically, this json should have a `action` key (with the name of the tool to use)
        and a `action_input` key (with the input to the tool going here).
        The only values that should be in the "action" field are: {[t.name for t in tools]}
        The $JSON_BLOB should only contain a SINGLE action, 
        do NOT return a list of multiple actions.
        Here is an example of a valid $JSON_BLOB:
        ```
        {{{{
            "action": $TOOL_NAME,
            "action_input": $INPUT
        }}}}
        ```
        The $JSON_BLOB must always be enclosed with triple backticks!

        ALWAYS use the following format:
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action:```
        $JSON_BLOB
        ```
        Observation: the result of the action... 
        (this Thought/Action/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin! Reminder to always use the exact characters `Final Answer` when responding.'
        """ 
        retriever_agent = AgentCoordinator.create_agent(lm, model_id, tools, retriever_prompt_template)
        retriever_node = functools.partial(agent_node, agent=retriever_agent, name="PaperInfoRetriever")

        self.workflow = StateGraph(AgentState)
        self.workflow.add_node("PaperInfoRetriever", retriever_node)
        self.workflow.add_node("supervisor", AgentCoordinator(lm))
        
        for member in AgentCoordinator.members:
            # We want our workers to ALWAYS "report back" to the supervisor when done
            self.workflow.add_edge(member, "supervisor")
            
        # The supervisor populates the "next" field in the graph state which routes to a node or finishes
        conditional_map = {k: k for k in AgentCoordinator.members}
        conditional_map["FINISH"] = END
        # print(f"Conditional Map: {conditional_map}")
        def get_next(state: AgentState):
            print(f"State: {state}")
            return state["next"]
        self.workflow.add_conditional_edges("supervisor", get_next, conditional_map)
        
        # Entrypoint
        self.workflow.add_edge(START, "supervisor")

        self.graph = self.workflow.compile()
    
    
    def __call__(self, text: str, stream_mode: StreamMode | list[StreamMode] = "values"):
        inputs = {"messages": [HumanMessage(content=text)]}
        config = RunnableConfig(recursion_limit=100)
        try:
            events = self.graph.stream(inputs, config, stream_mode=stream_mode)
        except GraphRecursionError:
            print("Graph recursion limit reached.")
            events = ["N/A"]
        return events
    
    
    def get_graph(self):
        return self.graph.get_graph()

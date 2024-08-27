import json
import operator
import os
import re
import pprint

from enum import Enum
from typing import Any, Dict, List, TypedDict, Annotated, Optional, Type
from langchain_huggingface import ChatHuggingFace
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import BaseTool, tool
from langchain_core.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolInvocation, ToolExecutor
# from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict
import sqlite3

from llm_foundation import logger
from llm_foundation.routing import ToolMaster
from llm_foundation.utils import banner, show_banner


@tool
def calculate(math_exp):
    """Performs arithmetic calculations of the mathematical expression math_exp.

    Args:
        math_exp: mathematical expression to be evaluated
    """    
    return eval(math_exp)

@tool
def average_dog_weight(breed_name):
    """Returns the average weight of a dog breed.

    Args:
        breed_name: the name of the dog breed
    """
    match breed_name.lower():
        case "scottish terrier": 
            return("Scottish Terriers average 20 lbs")
        case "border collie":
            return("a Border Collies average weight is 37 lbs")
        case "toy poodle":
            return("a toy poodle average weight is 7 lbs")
        case _:
            return("An average dog weights 50 lbs")


class MathExp(BaseModel):
    math_exp: Optional[str] = Field(description="mathematical expression")

class Calculator(BaseTool):
    
    name = "calculate"
    description: str = (
        "Performs arithmetic calculations of the mathematical expression math_exp. "
        "Input args is a mathematical expresion to be evaluated."
    )
    args_schema: Type[BaseModel] = MathExp
    return_direct: bool = True

    def _run(
        self, math_exp, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Any:
        """Used to evaluate mathematical expresions."""
        try:
            result = eval(math_exp)
        except:
            result = "Invalid calculation for expresion {}".format(math_exp)
        return result

    async def _arun(
        self, math_exp, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


class DogBreed(BaseModel):
    breed_name: Optional[str] = Field(description="breed_name")

class DogBreedWeighter(BaseTool):
    
    name = "average_dog_weight"
    description: str = (
        "Returns the average weight of a dog breed. "
        "Input args is a breed_name representing the name of the dog breed."
    )
    args_schema: Type[BaseModel] = DogBreed
    return_direct: bool = True

    def _run(
        self, breed_name, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Any:
        breed_name = breed_name.lower().strip()
        """Used to return the weight of a dog breed."""
        if breed_name in "scottish terrier": 
            return("a Scottish Terrier average 20 lbs")
        elif breed_name in "border collie":
            return("a Border Collie average weight is 37 lbs")
        elif breed_name in "toy poodle":
            return("a toy poodles average weight is 7 lbs")
        else:
            return("An average dog weights 50 lbs")

    async def _arun(
        self, breed_name, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


TOOLS = {
    "calculate": calculate,
    "average_dog_weight": average_dog_weight
}

langchain_tools = [calculate, average_dog_weight]
# langchain_tools = [Calculator(), DogBreedWeighter()]


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Suggestions(BaseModel):
    """Suggestions to the plan."""
    elems: List[str] #= Field(description="List of suggestions to improve the plan.")

class CandidateResponseValidator(BaseModel):
    """Validation for a candidate response."""
    validation: bool


class DoggieToolMaster(ToolMaster):
    
    @banner(text="Doggie Post Call Tool", level=2, mark_fn_end=False)
    def post_call_tool(self, state, responses):
        tool_messages = []
        evidences = []
        for response in responses:
            tool_call_id, tool_name, response = response
            tool_message = ToolMessage(
                content=str(response), name=tool_name, tool_call_id=tool_call_id
            )
            tool_messages.append(tool_message)
            evidences.append(response)

        return {
            "last_node": "call_tools",
            "messages": tool_messages,  # We return a list, because this will get added to the existing list of messages
            "evidences": state["evidences"] + evidences,
        }

class AgentState(TypedDict):
    task: str
    last_node: str
    plan: str
    plan_approved: bool
    suggestions: List[str]
    max_plan_revisions: int
    revision_number: int
    count: Annotated[int, operator.add]
    messages: Annotated[list, add_messages]
    evidences: List[str]
    candidate_response: str
    final_response: str
    
def _show_state_messages(messages: List[BaseMessage], as_text:bool = False):
    message_objects = {}
    for i, message in enumerate(messages):
        message_id = f"message_{i}"
        message_objects[message_id] = message.content
    if as_text:
        return pprint.pprint(message_objects, indent=2)
    return json.dumps(message_objects)

def show_state(state: AgentState, as_text:bool = False):
    json_state = json.dumps({
        "plan": state['plan'],
        "plan_approved": state['plan_approved'],
        "suggestions": state['suggestions'],
        "max_plan_revisions": state['max_plan_revisions'],
        "revision_number": state['revision_number'],
        "count": state['count'],
        "messages": _show_state_messages(state['messages'], as_text),
        "evidences": state['evidences'],
        "candidate_response": state['candidate_response'],
        "final_response": state['final_response']
    })
    if as_text:
        return pprint.pprint(state, indent=2)
    return json_state


class DoggieMultiAgent():

    def __init__(self, lm, name="Pepe"):
        
        self.name = name
        self.model = lm
        self.model = self.model.bind_tools(langchain_tools)
        self.tool_master = DoggieToolMaster(langchain_tools)
  
        
        self.PLAN_PROMPT = """
        You are an expert planner with knowledge and tools about all that has \
        do do with dogs and dog breeds, so you can be tasked to make a plan \
        using that knowledge manage to answer any user task of that topic. \
        Avoid any step for which you don't have a clear idea and avoid ask the user for clarification. \
        Think step by step. \
        
        Use the following format: \
        Question: the task below. \
        Thought: you should always think about what to do, do not use any tool if it is not needed. \
        Don't try to execute any step of the plan.\
        Plan: the depicted plan steps. \
        
        Utilize the suggestions below as needed to improve the plan: \
        {suggestions} \
            
        Always respond with a list of plan steps, avoiding directly calling a tool to solve the problem. \
        User task: """.strip()
        
        self.PLAN_CRITIC_PROMPT = """
        You are an expert decision maker grading an plan submission. \
        Return a list of critiques and suggestions for the user's plan submission. \
        The list will contain straight to the point recommendations on how to improve the plan, \
        like step clarification, decomposition in fine-grain steps, etc. \
        Be straight to the point and don't overcomplicate steps if the plan is already neat. \
        So if the plan is reasonable, just return a list of empty suggestions.""".strip()
        
        self.PLAN_EXECUTOR_PROMPT = """
        You are a plan executor tasked with finding the response to a user question following a plan. \
        Generate the best response possible for the user's request using tools if needed. \
        If the information is in the evidences, do not call any tool and use the evidence information \
        to generate the response. \
        Evidences: \
        {evidences} \
        ------\n \
        {content}"""

        self.RESPONSE_VALIDATOR_PROMPT = """
        You are an expert judge for validationg responses to questions. \
        Given the question and the candidate response below, return if \
        you agree or not that the response makes sense. Respond only with True or False. """.strip()
        
        # Define Agent graph nodes
        builder = StateGraph(AgentState)
        builder.add_node("planner", self.plan_node)
        builder.add_node("planner_critic", self.plan_critic)
        builder.add_node("plan_executor", self.plan_executor_node)
        builder.add_node("action", self.tool_master.agentic_tool_call)
        builder.add_node("response_validator", self.response_validator)
        
        # Define edges
        builder.add_edge("planner", "planner_critic")
        builder.add_conditional_edges(
            "planner_critic", 
            self.should_refine_plan, 
            {
                "review": "planner",  # Back to planner
                "plan_executor": "plan_executor", 
            }
        )
        
        builder.add_conditional_edges(
            "plan_executor", 
            self.should_continue, 
            {
                "response_validation": "response_validator",
                "reexecute_tool": "action",
                END: END, 
            }
        )

        builder.add_conditional_edges(
            "response_validator", 
            self.should_accept_response, 
            {
                "no": "plan_executor",
                "yes": END,
            }
        )

        
        builder.add_edge("action", "plan_executor")

        
        # Entry point for the workflow
        builder.set_entry_point("planner")
        
        memory = SqliteSaver(conn=sqlite3.connect(":memory:", check_same_thread=False))
        self.graph = builder.compile(
            checkpointer=memory,
            interrupt_after=['planner', "planner_critic"] # 'generate', 'reflect', 'research_plan', 'research_critique']
        )
        

    @banner(text="Plan Node")
    def plan_node(self, state: AgentState):
        logger.info(f"Current state:\n{show_state(state, as_text=True)}")
        possible_suggestions = "\n\n".join(state['suggestions'] or [])
        plan = self.PLAN_PROMPT.format(suggestions=possible_suggestions)
        plan = " ".join(plan.strip().split())
        logger.info(f"Plan prompt:\n{plan.strip()} ")
        # logger.info(f"{state['task']} ")
        messages = [
            SystemMessage(content=plan), 
            HumanMessage(content=state['task'])
        ]
        response = self.model.invoke(messages)
        logger.info(f"Depicted plan:\n{response}")
        return {"plan": response.content,
                "last_node": "planner",
                "revision_number": state.get("revision_number", 1) + 1,
                "count": 1,
        }

    @banner(text="Plan Critic Node")
    def plan_critic(self, state: AgentState):
        _show_state_messages(state["messages"], as_text=True)
        logger.info(f"Current plan:\n{state['plan']}")

        #  Local coding the schema directly (fails in getting the array properly, gets a string instead)
        #
        # suggestions_json_schema = {
        #     "title": "suggestions",
        #     "description": "Suggestions from the planner reviewer.",
        #     "type": "object",
        #     "properties": {
        #         "suggestions": {
        #             "type": "array",
        #             "description": "The list of strings, each one being a suggestion",
        #         },
        #     },
        #     "required": ["suggestions",],
        # }        
        # critic_response = self.model.with_structured_output(suggestions_json_schema).invoke([
        #     SystemMessage(content=self.PLAN_CRITIC_PROMPT),
        #     HumanMessage(content=state['plan'])
        # ])
        
        # Coding opeanai schema
        #        
        # client = instructor.from_openai(
        #     OpenAI(
        #         base_url="http://localhost:11434/v1",
        #         api_key="ollama",  # required, but unused
        #     ),
        #     mode=instructor.Mode.JSON,
        # )
        # critic_response = client.chat.completions.create(
        #     model="llama3.1",
        #     messages=[
        #         {
        #             "role": "system",
        #             "content": self.PLAN_CRITIC_PROMPT,
        #         },
        #         {
        #             "role": "user",
        #             "content": state['plan'],
        #         },                
        #     ],
        #     response_model=Suggestions,
        # )

        critic_response = self.model.with_structured_output(Suggestions).invoke([
            SystemMessage(content=self.PLAN_CRITIC_PROMPT),
            HumanMessage(content=state['plan'])
        ])
        pprint.pprint(f"Response ({critic_response})")
        
        plan_approved = len(critic_response.elems) == 0
        show_banner(f"Plan Approved: {plan_approved}", level=4)
        return {
            "plan_approved": plan_approved,
            "suggestions": critic_response.elems,
            "last_node": "planner_critic",
            "count": 1,
        }

    @banner(text="Plan Executor Node")
    def plan_executor_node(self, state: AgentState):
        _show_state_messages(state["messages"], as_text=True)
        logger.info(f"Current plan:\n{state['plan']}")
        content = state['plan']
        evidences = "\n\n".join(state['evidences'] or [])
        messages = [
            SystemMessage(
                content=self.PLAN_EXECUTOR_PROMPT.format(content=content, evidences=evidences)
            ),
            HumanMessage(
                content=f"Here is the task:\n\n{state['task']}"
            ),
            ]
        logger.info(f"Messages: {messages}")
        response = self.model.invoke(messages)
        logger.info(f"Response: {response}")
        return {
            "messages": [response],
            "last_node": "plan_executor",
            "count": 1,
            "candidate_response": response.content if not response.tool_calls else "",
        }

    @banner(text="Response Validator Node")
    def response_validator(self, state):
        logger.info(show_state(state, as_text=True))
        candidate_response = state["candidate_response"]
        messages = [
            SystemMessage(
                content=self.RESPONSE_VALIDATOR_PROMPT
            ),
            HumanMessage(
                content=f"Question: {state['task']}\n\nResponse: {candidate_response}"
            ),
            ]
        
        response = self.model.with_structured_output(CandidateResponseValidator).invoke(messages)
        print(f"Validator response: {response}")
        return {
            "last_node": "response_validator",
            'final_response': candidate_response if response.validation else "",
            'candidate_response': '',
        }

    @banner(text="Should Refine Plan Decision Point", level=4)
    def should_refine_plan(self, state):
        logger.info(f"Current state:\n{show_state(state, as_text=True)}")
        if not state["plan_approved"] and state["revision_number"] < state["max_plan_revisions"]:
            show_banner(f"Refining plan! Plan Approved {state['plan_approved']} Revision Number {state['revision_number']} out of {state['max_plan_revisions']}")
            return "review"
        show_banner(f"Continuing with Plan Executor! Plan Approved {state['plan_approved']} Revision Number {state['revision_number']} out of {state['max_plan_revisions']}")
        return "plan_executor"
    
    @banner(text="Should Continue Plan Decision Point", level=4)
    def should_continue(self, state):
        print(f"State: {state}")
        messages = state["messages"]
        print(f"Messages: {messages}")
        last_message = messages[-1]
        print(f"-----Last Message: {last_message}----")
        # If there is no function call, then we finish
        
        if not last_message.tool_calls:
            if state['candidate_response'] != "":
                show_banner("No function call, but candidate response", level=3)
                return "response_validation"
            show_banner("No function call, finishing", level=3)
            return END
        else:
            show_banner("Function call, reexecuting", level=3)
            return "reexecute_tool"

    @banner(text="Should Accept Response Decision Point", level=4)
    def should_accept_response(self, state):
        resp = "no" if state['final_response'] == "" else "yes"
        print(f"-------------- Should Accept Response ({resp}) ---------------")
        return resp

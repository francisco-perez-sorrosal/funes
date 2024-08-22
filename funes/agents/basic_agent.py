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
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import BaseTool, tool
from langchain_core.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolInvocation, ToolExecutor
# from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict
import sqlite3
import instructor
from openai import OpenAI

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


# class Suggestions(TypedDict):
#     """Suggestions to the plan."""
#     elems: Annotated[List[str], ..., "List of suggestions to improve the plan."]


class AgentState(TypedDict):
    task: str
    lnode: str
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

class DoggieMultiAgent():

    def __init__(self, lm, name="Pepe"):
        
        self.name = name
        self.model = lm
        self.model = self.model.bind_tools(langchain_tools)
        self.tool_executor = ToolExecutor(langchain_tools)
  
        
        self.PLAN_PROMPT = """
        You are an expert planner with knowledge and tools about all that has \
        do do with dogs and dog breeds, so you can be tasked to make a plan \
        using that knowledge manage to answer any user task of that topic. \
        Avoid any step for which you don't have a clear idea and avoid ask the user for clarification. \
        Think step by step. \
        Utilize the suggestions below as needed to improve the plan: \
        {suggestions} \
        Just respond with the plan steps. \
        User task: """.strip()
        
        self.PLAN_CRITIC_PROMPT = """
        You are an expert decision maker grading an plan submission. \
        Return a list of critiques and suggestions for the user's plan submission. \
        The list will contain straight to the point recommendations on how to improve the plan, \
        like step clarification, decomposition in fine-grain steps, etc. \
        Be straight to the point and don't overcomplicate steps. \
        If the plan already seems reasonable just return a list of empty suggestions.""".strip()
        
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
        builder.add_node("action", self.call_tool)
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

        
    def plan_node(self, state: AgentState):
        pprint.pprint("-------------- Plan Node ---------------")
        possible_suggestions = "\n\n".join(state['suggestions'] or [])
        plan = self.PLAN_PROMPT.format(suggestions=possible_suggestions)
        pprint.pprint(f"Plan prompt:\n{plan} ")
        pprint.pprint(f"{state['task']} ")
        messages = [
            SystemMessage(content=plan), 
            HumanMessage(content=state['task'])
        ]
        response = self.model.invoke(messages)
        pprint.pprint(f"Response:\n{response.content}")
        pprint.pprint("-------------- End Plan Node ---------------")
        return {"plan": response.content,
                "lnode": "planner",
                "revision_number": state.get("revision_number", 1) + 1,
                "count": 1,
        }

    def plan_critic(self, state: AgentState):
        
        pprint.pprint("-------------- Plan Critic Node ---------------")        
        pprint.pprint(state['plan'])

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
        pprint.pprint(f"-------------- End Plan Critic Node (Approved: {plan_approved}) ---------------")
        return {
            "plan_approved": plan_approved,
            "suggestions": critic_response.elems,
            "lnode": "planner_critic",
            "count": 1,
        }

    def plan_executor_node(self, state: AgentState):
        print("-------------- Plan Executor Node ---------------")
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
        print(f"Messages: {messages}")
        response = self.model.invoke(messages)
        print(f"Response: {response}")
        print("-------------- End Plan Executor Node ---------------")
        return {
            "messages": [response],
            "lnode": "plan_executor",
            "count": 1,
            "candidate_response": response.content if not response.tool_calls else "",
        }
        
    # Define the function to execute tools
    def call_tool(self, state):
        print("-------------- Call Tool ---------------")
        messages = state["messages"]
        # Based on the continue condition
        # we know the last message involves a function call
        last_message = messages[-1]
        # We construct an ToolInvocation from the function_call
        tool_call = last_message.tool_calls[0]
        action = ToolInvocation(
            tool=tool_call["name"],
            tool_input=tool_call["args"],
        )
        # We call the tool_executor and get back a response
        response = self.tool_executor.invoke(action)
        print(f"Response: {response}")
        # We use the response to create a ToolMessage
        tool_message = ToolMessage(
            content=str(response), name=action.tool, tool_call_id=tool_call["id"]
        )
        print(f"Tool Message: {tool_message}")
        print("-------------- End Call Tool ---------------")
        # We return a list, because this will get added to the existing list
        return {
            "lnode": "call_tool",
            "messages": [tool_message],
            "evidences": state["evidences"] + [response],
        }

    def response_validator(self, state):
        print("-------------- Response Validator ---------------")
        print(state)
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
            "lnode": "response_validator",
            'final_response': candidate_response if response.validation else "",
            'candidate_response': '',
        }

    def should_refine_plan(self, state):
        print("-------------- Should Refine Plan ---------------")
        print(f"State: {state}")
        if not state["plan_approved"] and state["revision_number"] < state["max_plan_revisions"]:
            print(f"Refining plan! Plan Approved {state['plan_approved']} Revision Number {state['revision_number']}({state['max_plan_revisions']})")
            return "review"
        return "plan_executor"
    
    def should_continue(self, state):
        print("-------------- Should Continue ---------------")
        print(f"State: {state}")
        messages = state["messages"]
        print(f"Messages: {messages}")
        last_message = messages[-1]
        print(f"-----Last Message: {last_message}----")
        # If there is no function call, then we finish
        
        if not last_message.tool_calls:
            if state['candidate_response'] != "":
                print("----- No function call, but candidate response -----")
                return "response_validation"
            print("----- No function call, finishing -----")
            return END
        else:
            print("----- Function call, reexecuting -----")
            return "reexecute_tool"

    def should_accept_response(self, state):
        resp = "no" if state['final_response'] == "" else "yes"
        print(f"-------------- Should Accept Response ({resp}) ---------------")
        return resp
        

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
    if breed_name in "Scottish Terrier": 
        return("Scottish Terriers average 20 lbs")
    elif breed_name in "Border Collie":
        return("a Border Collies average weight is 37 lbs")
    elif breed_name in "Toy Poodle":
        return("a toy poodle average weight is 7 lbs")
    else:
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
            return("Scottish Terriers average 20 lbs")
        elif breed_name in "border collie":
            return("a Border Collies average weight is 37 lbs")
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

# class Suggestions(TypedDict):
#     """Suggestions to the plan."""
#     elems: Annotated[List[str], ..., "List of suggestions to improve the plan."]


class AgentState(TypedDict):
    task: str
    lnode: str
    plan: str
    plan_approved: bool
    suggestions: List[str]
    max_revisions: int
    revision_number: int
    # draft: str
    # critique: str
    # content: List[str]
    # queries: List[str]
    count: Annotated[int, operator.add]

class DoggieMultiAgent():
    def __init__(self, lm, name="Pepe"):
        
        self.name = name
        self.model = lm
        self.model.bind_tools(langchain_tools)
        
        # self.PLAN_PROMPT = """
        # You are an expert planner knowledgeable about all that has do do with dogs and dog breeds \
        # , so you can be tasked to make a plan to manage to answer any question related to that. \
        # For example you can be tasked with the task to calculate the average weight of one or several dogs of \
        # different breeds, so you can be come up with plan consisting of a series of steps similar to these: \
        # 1. Identify the breed of the dog or dogs that are involved in the question. \
        # 2. Calculate the average weight of each dog depending on its breed. \
        # 3. Perform the math calculations to get the final result. \
        # 4. Reflect on the results and come up with a critique the plan if necessary. \
        # 5. Execute the critique plan if you can't find a satisfactory response. Otherwise deliver what you to the user. \
        # If the user provides suggestions, respond with a revised version of your previous attempts. \
        # Utilize all the information below as needed: \
        # {suggestions}""".strip()
        
        self.PLAN_PROMPT = """
        You are an expert planner with knowledge and tools about all that has \
        do do with dogs and dog breeds, so you can be tasked to make a plan \
        using that knowledge and tools to manage to answer any user task of that topic. \
        Think step by step. \
        Tools available: \
        "average_dog_weight": Returns the average weight of a dog breed. \
        Utilize the suggestions below as needed to improve the plan: \
        {suggestions} \
        Just respond with the plan steps.. \
        User task: """.strip()
        
        self.PLAN_CRITIC_PROMPT = """
        You are an expert decision maker grading an plan submission. \
        Return a list of critiques and suggestions for the user's plan submission. \
        The list will contain straight to the point recommendations on how to improve the plan, \
        like step clarification, decomposition in fine-grain steps, etc. \
        Be straight to the point and don't overcomplicate steps. \
        If the plan already seems reasonable just return a list of empty suggestions.""".strip()
        
        # Define Agent graph nodes
        builder = StateGraph(AgentState)
        builder.add_node("planner", self.plan_node)
        builder.add_node("planner_critic", self.plan_critic)
        
        # Define edges
        builder.add_edge("planner", "planner_critic")
        builder.add_conditional_edges(
            "planner_critic", 
            self.should_refine_plan, 
            {
                "review": "planner",  # Back to planner
                END: END, 
            }
        )
        
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

    def should_refine_plan(self, state):
        if not state["plan_approved"] and state["revision_number"] < state["max_revisions"]:
            print(f"Refining plan! Plan Approved {state['plan_approved']} Revision Number {state['revision_number']}({state['max_revisions']})")
            return "review"
        return END

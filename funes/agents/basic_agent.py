import operator
import os
import re

from enum import Enum
from typing import Any, Dict, TypedDict, Annotated, Optional, Type
from langchain_huggingface import ChatHuggingFace
from langchain_openai import ChatOpenAI
from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import BaseTool, tool
from langchain_core.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langgraph.graph import StateGraph, END
from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel

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

# langchain_tools = [calculate, average_dog_weight]
langchain_tools = [Calculator(), DogBreedWeighter()]


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

def msg_builder(content:str, role:Role = Role.USER):
    return {"role": role, "content": content}

action_re = re.compile('^Action: (\w+): (.*)$') 


class BasicAgent:
    def __init__(self, llm, system_msg:str = ""):
        self.llm = llm
        self.chat = ChatHuggingFace(llm=self.llm, model_id="meta-llama/Meta-Llama-3-8B-Instruct")
        self.system_msg = system_msg
        self.messages = []
        if self.system_msg:
            self.messages.append(msg_builder(self.system_msg, Role.SYSTEM))
                                 
    def __call__(self, msg):
        self.messages.append(msg_builder(msg))
        result = self.execute()
        self.messages.append(msg_builder(result, Role.ASSISTANT))
        return result
    
    def execute(self):
        completion = self.chat.invoke(self.messages)
        print(completion.content)
        return completion.content

        # completion = self.llm.chat.completions.create(
        #     model="gpt-4o",
        #     temperature=0,
        #     messages=self.messages,
        # )
        # return completion.choices[0].message.content


    def query(self, question, known_tools: Dict[str, str], max_turns=5):
        i = 0
        next_prompt = question
        result = ""
        while i < max_turns:
            i += 1
            result = self(next_prompt)
            print("JFSDJFD")
            print(result)
            actions = [
                action_re.match(a) 
                for a in result.split('\n') 
                if action_re.match(a)
            ]
            if actions:
                # There is an action to run
                action, action_input = actions[0].groups()
                if action not in known_tools:
                    raise Exception("Unknown action: {}: {}".format(action, action_input))
                print(" -- running {} {}".format(action, action_input))
                observation = known_tools[action](action_input)
                print("Observation:", observation)
                next_prompt = "Observation: {}".format(observation)
            else:
                if result and "Answer" in result:
                    return result
                else:
                    return "No result found"

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    
    
class BasicLGAgent:

    def __init__(self, model, tools, system="", memory=None,name="Pepe"):
        self.system = system
        self.name = name
        graph = StateGraph(AgentState)
        
        graph.add_node("llm", self.call_llm)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(
            checkpointer=memory,
            interrupt_before=["action"]
        )
        self.tools = {t.name: t for t in tools}
        # self.tools = tools
        print(f"Tools: {self.tools}")
        print(f"Model: {type(model)}")
        if isinstance(model, HuggingFaceEndpoint):
            chat = ChatHuggingFace(llm=model, model_id="meta-llama/Meta-Llama-3-8B-Instruct")
            self.model = chat.bind_tools(tools)
            print(f"Chat model {self.model.model_id} for agent created")            
        else:
            print("Non HF Endpoint")
            self.model = model.bind_tools(tools)
        print(f"Model: {self.model.name}")

    def exists_action(self, state: AgentState):
        print("In exists action")
        result = state['messages'][-1]
        print(f"Exist Action {len(result.tool_calls) > 0}")
        return len(result.tool_calls) > 0

    def call_llm(self, state: AgentState):
        messages = state['messages']
        if isinstance(messages[-1], ToolMessage) and messages[-1].content != "":
            print("Tool Message with answer found. Adding Facts")
            system = self.system.format(facts=messages[-1].content)
            messages = [SystemMessage(content=system)] + messages
        else:            
            if self.system:
                messages = [SystemMessage(content=self.system.format(facts=""))] + messages
        self.print_stream(messages)
        print("============= Calling model =============")
        message = self.model.invoke(messages)
        print(f"============= Model response:\n{message}\n=============")
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        print("---- Taking action ----")
        print(state)
        tool_calls = state['messages'][-1].tool_calls
        print(tool_calls)
        print("---- Action taken -----")
        results = []
        for t in tool_calls:
            print(f"======= Calling: {t['name']} =====")
            if not t['name'] in self.tools:      # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}
    
    def print_stream(self, stream):
        for message in stream:            
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()

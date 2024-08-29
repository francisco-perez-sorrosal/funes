from typing import Any, Dict, List, TypedDict, Annotated, Optional, Type
from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import BaseTool, tool
from langchain_core.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
# from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field

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
    
    name = "calculator"
    description: str = (
        "Performs arithmetic calculations of the mathematical expression math_exp. "
        "Input args is a mathematical expresion to be evaluated (e.g. 2 + 2)."
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
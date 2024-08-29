import json
import functools
import operator
import os
import re
import pprint
from enum import Enum
from typing import Any, Dict, List, TypedDict, Annotated, Optional, Type
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain.prompts import ChatPromptTemplate
# from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict
import sqlite3

from llm_foundation import logger
from llm_foundation.routing import ToolMaster
from llm_foundation.utils import banner, show_banner
from llm_foundation.extractors import PlanExtractor, Plan, MultiTreePlan
from funes.agents.tools import Calculator, DogBreedWeighter, DoggieToolMaster, calculate, average_dog_weight
from funes.agents.worker_agent import WorkerAgent, WorkerAgentState

TOOLS = {
    "calculator": calculate,
    "average_dog_weight": average_dog_weight
}

# langchain_tools = [calculate, average_dog_weight]
langchain_tools = [Calculator(), DogBreedWeighter()]


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
        logger.info("Adding Plan Extractor to the model")
        self.plan_extractor_tool = PlanExtractor.build_tool(lm, Plan, use_pydantic_output=True)
        langchain_tools.append(self.plan_extractor_tool)
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
        Plan: the depicted plan steps. Group all the logic substeps related to the same step together. \
        
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
        # builder.add_node("action", self.tool_master.agentic_tool_call)
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
        
        builder.add_edge("plan_executor", "response_validator")
        # builder.add_conditional_edges(
        #     "plan_executor", 
        #     self.should_continue, 
        #     {
        #         "response_validation": "response_validator",
        #         "reexecute_tool": "action",
        #         END: END, 
        #     }
        # )

        builder.add_conditional_edges(
            "response_validator", 
            self.should_accept_response, 
            {
                "no": "plan_executor",
                "yes": END,
            }
        )
        
        # builder.add_edge("action", "plan_executor")
        
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
        current_plan = state['plan']
        logger.info(f"Current plan:\n{current_plan}")

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are skilled decision maker and plan extractor able to identify hierarchical complex plan steps in plain text."),
            ("user", "Extract the plan from the following text: {input}"),
        ])        
        logger.info("++++++++++++++++++++++++++++++++++")        
        chain = prompt | self.model | self.tool_master
        chain_response = chain.invoke({'input': current_plan})
        plan = chain_response[0][2]
        for i, step in enumerate(plan.plan):
            logger.info(f"Plan Step {i}: {step}\n")
            
        multi_tree_plan = PlanExtractor.build_multi_tree_plan(plan)
        
        logger.info(multi_tree_plan.print_multi_tree())
                
        leaf_sorted_plan = multi_tree_plan.traverse_leafs_first_dependants()        
        show_banner(f"Leaf sorted plan:\n{leaf_sorted_plan}", level=4)
        
        from langchain_core.runnables import RunnableConfig
        
        collective_evidences: Dict[str, list] = {}
        candidate_respoonses = []
        for step in leaf_sorted_plan:
            configurable = {
                    "thread_id": f"worker_thr_{step.id}",
            }
            worker_agent_config = RunnableConfig(recursion_limit=3, configurable=configurable)
            worker_agent = WorkerAgent(self.model, f"Worker_{step.id}")
            initial_agent_state = {
                'task': step,
                'response': '',
    
            }
            logger.warning(f"Invoking graph with state: {initial_agent_state}")
            try:
                dependent_evidences = []
                if step.is_root() or (step.is_root() and step.is_leaf()):
                    show_banner(f"Delegating root step {step.id} on Working Agent: {step.description}", level=2)
                    for s in step.depending_steps:
                        dependent_evidences.append(collective_evidences[s.id])                        
                else:
                    if step.is_leaf():
                        show_banner(f"Delegating leaf step {step.id} on Working Agent: {step.description}", level=2)
                    else:
                        show_banner(f"Delegating intermediate step {step.id} on Working Agent: {step.description}", level=2)
                        for s in step.depending_steps:
                            dependent_evidences.append(collective_evidences[s.id])

                logger.info(f"Dependent evidences: {dependent_evidences}")
                initial_agent_state['evidences'] = dependent_evidences
                response = worker_agent.graph.invoke(initial_agent_state, worker_agent_config)
                collective_evidences[step.id] = response['response']
                if step.is_root() or (step.is_root() and step.is_leaf()):
                    candidate_respoonses.append(response["response"])
                
            except GraphRecursionError as e:
                logger.warning(f"Step {step.id} graph recursion error: {e}")
                response = None
                
        final_response = " ".join(candidate_respoonses)
        logger.info(f"Worker agent response: {final_response}")

        return {
            "messages": [response],
            "last_node": "plan_executor",
            "count": 1,
            "candidate_response": AIMessage(final_response),
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

    # Decision points
    @banner(text="Should Refine Plan Decision Point", level=4)
    def should_refine_plan(self, state):
        logger.info(f"Current state:\n{show_state(state, as_text=True)}")
        logger.info(f"Plan Approved {state['plan_approved']} Rev.# {state['revision_number']} / {state['max_plan_revisions']}")
        if not state["plan_approved"] and state["revision_number"] < state["max_plan_revisions"]:
            show_banner(f"DECISION: Refine plan!", level=4) 
            return "review"
        show_banner(f"DECISION: Continuing with Plan Executor!", level=4)
        return "plan_executor"
    
    @banner(text="Should Continue Plan Decision Point", level=4)
    def should_continue(self, state):
        logger.info(f"Current state:\n{show_state(state, as_text=True)}")
        messages = state["messages"]
        last_message = messages[-1]
        logger.debug(f"-----Last Message-----\n{last_message}")
        # If there is no function call, then we finish        
        if not last_message.tool_calls:
            if state['candidate_response'] != "":
                show_banner(f"DECISION: No function call, but candidate response!", level=4)                
                return "response_validation"
            show_banner(f"DECISION: No function call, finishing!", level=4)
            return END
        else:
            show_banner(f"DECISION: Function call, re-executing!", level=4)
            return "reexecute_tool"

    @banner(text="Should Accept Response Decision Point", level=4)
    def should_accept_response(self, state):
        logger.info(f"Current state:\n{show_state(state, as_text=True)}")
        resp = "no" if state['final_response'] == "" else "yes"
        show_banner(f"DECISION: Should Accept Response ({resp})", level=4)
        return resp

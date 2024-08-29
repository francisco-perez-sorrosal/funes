from typing import List, Tuple, TypedDict

from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain.prompts import ChatPromptTemplate

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from llm_foundation.extractors import PlanExtractor, Plan

from funes.agents.tools import Calculator, DogBreedWeighter, DoggieToolMaster
from llm_foundation import logger
from llm_foundation.utils import banner, show_banner
from llm_foundation.extractors import Step

import sqlite3


class WorkerAgentState(TypedDict):
    task: Step
    response: str
    evidences: List[str]

class WorkerCandidateResponseValidator(BaseModel):
    """Validation for a worker candidate response."""
    validation: bool


class WorkerAgent():
    
    def __init__(self, lm, name="Worker Agent"):
        
        self.name = name
        self.model = lm
        tools = [Calculator(), DogBreedWeighter()]
        self.plan_extractor_tool = PlanExtractor.build_tool(lm, Plan, use_pydantic_output=True)
        tools.append(self.plan_extractor_tool)

        self.tool_master = DoggieToolMaster(tools)
        
        self.WORKER_PROMPT = """
        You are an expert and efficient worker agent that will execute the tasks received from other agents
        """.strip()
        
        self.WORKER_RESPONSE_VALIDATOR_PROMPT = """
        You are an expert judge for validationg responses to questions. \
        Given a user task/question and evidences, assess the candidate response below; return whether \
        you agree or not that the response fulfills/answers task/question. Respond only with True or False. """.strip()

        # Define Agent graph nodes
        builder = StateGraph(WorkerAgentState)
        builder.add_node("task_executor", self.task_executor)
        builder.add_node("worker_response_validator", self.worker_response_validator)
        
        builder.add_edge("task_executor", "worker_response_validator")
        builder.add_conditional_edges(
            "worker_response_validator", 
            self.worker_should_accept_response, 
            {
                "no": "task_executor",
                "yes": END,
            }
        )
        
        # Entry point for the worker agent workflow
        builder.set_entry_point("task_executor")
        
        memory = SqliteSaver(conn=sqlite3.connect(":memory:", check_same_thread=False))
        self.graph = builder.compile(
            checkpointer=memory,
        )        

    @banner(text="Task Executor Node")
    def task_executor(self, state: WorkerAgentState):
        
        task: Step = state["task"]
        evidences = "\n".join(state["evidences"])
        logger.info(f"Current task:\n{task.description}")
        logger.info(f"Current task:\n{evidences}")
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.WORKER_PROMPT),
            ("user", "Execute this task: {input}, using these facts {evidences}"),
        ])
        
        logger.info("++++++++++++++++++++++++++++++++++")
        chain = prompt | self.model | self.tool_master
        chain_list_of_responses = chain.invoke(
            {
                'input': task.description,
                'evidences': evidences,
            }
        )
        
        aggregate_response = ""
        for response in chain_list_of_responses:
            if isinstance(response, Tuple):
                _,_,response = response
                aggregate_response += str(response) + "\n"
            else:
                logger.warning(f"Response is not a tuple: {type(response)}")

        logger.info(f"Task execution response: {aggregate_response}")
                
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert summarizer. Be straight to the point with the task that follows."),
            ("user", "Given a task/query, its evidences and a response, come up with a sentence summarizing the response contextualized with the query. Task/Query: {input} Evidences {evidences} Response: {response}"),
        ])        


        summary_chain = summary_prompt | self.model
        response = summary_chain.invoke(
            {
                'input': task.description,
                'evidences': evidences,
                'response': aggregate_response,
            }
        )
        
        logger.info(f"Task execution response (after summary): {response.content}")
        state["response"] = response.content
        
        return state

    
    @banner(text="Worker Response Validator Node")
    def worker_response_validator(self, state):
        logger.info(f"Current state:\n{state}")
        
        step: Step = state["task"]
        question = step.description
        candidate_response = state["response"]
        evidences = state["evidences"]
        messages = [
            SystemMessage(
                content=self.WORKER_RESPONSE_VALIDATOR_PROMPT
            ),
            HumanMessage(
                content=f"Task/Question: {question} Evidences: {evidences} does the Response: {candidate_response} make sense?"
            ),
        ]
        print(f"Query: {question}")
        print(f"Evidences: {evidences}")
        print(f"Candidate Response: {candidate_response}")
        
                
        response = self.model.with_structured_output(WorkerCandidateResponseValidator).invoke(messages)
        logger.info(f"Worker Validation response: {response.validation}")
        return {            
            'response': candidate_response if response.validation else "",
        }
        
    @banner(text="Worker Should Accept Response Decision Point", level=4)
    def worker_should_accept_response(self, state):
        
        logger.info(f"Current state:\n{state}")
        resp = "no" if state['response'] == "" else "yes"
        show_banner(f"DECISION: Should Accept Response from Worker ({resp})", level=4)
        return resp

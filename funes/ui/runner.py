from collections.abc import Generator

from enum import Enum
from typing import Any, Optional, Tuple, Union
from pydantic import BaseModel


class RunnerStateType(int, Enum):
    ITER = 0
    STOP = 1
    MAX_ITERS = 2
    END = 3


class RunnerState(BaseModel):
    type: RunnerStateType
    thread_id: Optional[int] = None
    last_node: Optional[str] = None
    next_node: Optional[Tuple[str]] = None
    message: Optional[str] = None
    rev: Optional[int] = None
    acount: Optional[int] = None

class AgentRunner():

    def __init__(self, agent, max_iterations=10, share=False):
        self.agent = agent
        self.share = share
        self.partial_message = ""
        self.response = {}
        self.max_iterations = max_iterations
        self.iterations = []
        self.threads = []
        self.threads_started = {}
        self.thread_id = -1
        self.thread = {"configurable": 
            {"thread_id": str(self.thread_id)}
        }
        print(f"AgentRunner created for agent {self.agent.name}")


    def get_agent_name(self) -> str:
        return self.agent.name

    def get_display_state(self,):
        current_state = self.agent.graph.get_state(self.thread)
        lnode = current_state.values["lnode"]
        acount = current_state.values["count"]
        rev = current_state.values["revision_number"]
        nnode = current_state.next
        # print(lnode,nnode,self.thread_id,rev,acount)
        return lnode, nnode, self.thread_id, rev,acount


    def get_agent_state(self, key):
        current_values = self.agent.graph.get_state(self.thread)
        print(f"Getting agent state for key: {key} Current values: {current_values.values}")
        if key in current_values.values:
            lnode, nnode, self.thread_id, rev, astep = self.get_display_state()
            new_label = f"last_node: {lnode}, thread_id: {self.thread_id}, rev: {rev}, step: {astep}"
            return new_label, current_values.values[key]
        else:
            return "", None

    def is_started(self):
        return self.threads_started.get(self.thread_id, False)

    def run_agent(self, start: bool, topic: str, stop_after: list = []) -> Union[Generator[Any, Any, Any], RunnerState]:
        #global partial_message, thread_id,thread
        #global response, max_iterations, iterations, threads
        if start:
            print(f"Starting agent {self.get_agent_name()}")
            self.iterations.append(0)
            config = {'task': topic, 
                    'lnode': "", 
                    'planner': "no plan",
                    'revision_number': 0,
                    'max_revisions': 1,
                    'count':0}
            self.thread_id += 1  # new agent, new thread
            self.threads.append(self.thread_id)
            self.threads_started[self.thread_id] = True
        else:
            config = None  # This means continue execution when calling "invoke" below
        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}
        while self.iterations[self.thread_id] < self.max_iterations:
            print("*" * 100)
            print(f"Agent {self.get_agent_name()} (Thread: {self.thread_id} - Iter: {self.iterations[self.thread_id]} / {self.max_iterations})")
            print("*" * 100)
            self.response = self.agent.graph.invoke(config, self.thread)
            self.iterations[self.thread_id] += 1
            self.partial_message += str(self.response)
            self.partial_message += f"\n------------------\n\n"
            lnode, nnode, _, rev, acount = self.get_display_state()
            
            
            # print(f"After yield")
            config = None
            if not nnode:  
                print("Hit the end")
                yield RunnerState(type=RunnerStateType.END)
            elif lnode in stop_after:
                print(f"stopping due to stop_after {lnode} -> {stop_after}")
                yield RunnerState(type=RunnerStateType.STOP)
            else:
                print(f"Not stopping on lnode {lnode} ")
                print(f"thread_id: {self.thread_id},last_node: {lnode}, next_node: {nnode}, rev: {rev}, acount: {acount}")
                yield RunnerState(type=RunnerStateType.ITER,
                                thread_id=self.thread_id, 
                                last_node=lnode, 
                                next_node=nnode if len(nnode) != 0 else None, 
                                message=self.partial_message, 
                                rev=rev, 
                                acount=acount
                )
        yield RunnerState(type=RunnerStateType.MAX_ITERS)
    
    def __repr__(self):
        return f"AgentRunner(agent={self.agent}, max_iterations={self.max_iterations}, share={self.share})"

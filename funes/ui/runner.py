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
    
    class Agent_Thread():
        def __init__(self, id: int, agent_state: dict):
            self.id: int = id
            self.iterations = 0
            self.agent_state = agent_state
            self.config = {"configurable": 
                {"thread_id": str(self.id)}
            }
            print(f"Starting new thread {self.id} for agent")
        
        def is_started(self):
            return self.iterations > 0
        
        def inc_iterations(self):
            self.iterations += 1
            
        def get_agent_state(self) -> dict:
            return self.agent_state
        
        def __repr__(self):
            return f"Agent_Thread(id={self.id}, config={self.config}, iterations={self.iterations}, agent_state={self.agent_state})"
        
    
    def _new_thread(self, agent_state: dict) -> Agent_Thread:
        thread = AgentRunner.Agent_Thread(self.thread_id_counter, agent_state)
        self.thread_id_counter += 1
        self.threads[thread.id] = thread
        return thread

    def __init__(self, agent, max_iterations=10, share=False):
        self.thread_id_counter = 0
        self.agent = agent
        self.share = share
        self.partial_message = ""
        self.response = {}
        self.max_iterations = max_iterations
        self.iterations = []
        self.threads = {}
        self.current_thread: Optional[AgentRunner.Agent_Thread] = None
    #     self.thread_id = -1
    #     self.thread = {"configurable": 
    #         {"thread_id": str(self.thread_id)}
    #     }
        print(f"AgentRunner created for agent {self.agent.name}")


    def get_agent_name(self) -> str:
        return self.agent.name

    def get_display_state(self,):
        current_thread = self._get_current_thread()
        if not current_thread:
            print("No current thread!!!!")
            thread_config = {"configurable": 
                {"thread_id": str(-1)}
            }
        else:
            thread_config = current_thread.config
        
        current_state = self.agent.graph.get_state(thread_config)
        lnode = current_state.values["lnode"]
        acount = current_state.values["count"]
        rev = current_state.values["revision_number"]
        nnode = current_state.next
        # print(lnode,nnode,self.thread_id,rev,acount)
        return lnode, nnode, current_thread.id if current_thread else -1, rev,acount


    def get_agent_state(self, key):
        print("------------------------------------")
        current_thread = self._get_current_thread()
        print(f"Key: {key}")
        print(f"Current thread: {current_thread}")
        if not current_thread:
            thread_config = {"configurable": 
                {"thread_id": str(-1)}
            }
        else:
            thread_config = current_thread.config
        current_values = self.agent.graph.get_state(thread_config)
        print(f"Type current_values: {type(current_values)}")
        print(f"Getting agent state for key: {key} Current values: {current_values.values}")
        if key in current_values.values:
            lnode, nnode, thread_id, rev, astep = self.get_display_state()
            new_label = f"last_node: {lnode}, thread_id: {thread_id}, rev: {rev}, step: {astep}"
            return new_label, current_values.values[key]
        else:
            return "", None

    def is_started(self):
        return False if self._get_current_thread() is None else self._get_current_thread().is_started()
    
    def _get_current_thread(self) -> Optional[Agent_Thread]:
        return self.threads[self.current_thread.id] if self.current_thread else None

    def set_current_thread(self, thread_id: Optional[int] = None):
        print("X" * 100)
        print(f"Threads: {self.threads}")
        print(f"Setting current thread to {thread_id}")
        self.current_thread = self.threads[thread_id] if thread_id is not None else None
        print(f"Current thread set: {self.current_thread}")
        print("X" * 100)

    def get_thread_ids(self):
        return list(self.threads.keys())

    def current_thread_id(self):
        current_thread = self._get_current_thread()
        return -1 if not current_thread else current_thread.id
            
    def current_thread_config(self):
        current_thread = self._get_current_thread()
        print(f"Getting current Thread config for thread: {current_thread}")
        if not current_thread:
            current_thread_config = {"configurable": 
                {"thread_id": str(-1)}
            }
        else:
            current_thread_config = current_thread.config
        return current_thread_config

    def run_agent(self, topic: str, max_plan_revisions: int = 3, stop_after: list = []) -> Union[Generator[Any, Any, Any], RunnerState]:

        current_thread = self._get_current_thread()
        if current_thread is None:
            initial_agent_state = {
                'task': topic, 
                'lnode': "", 
                'planner': "no plan",
                'revision_number': 0,
                'max_plan_revisions': max_plan_revisions,
                'count':0
            }
            current_thread = self._new_thread(initial_agent_state)
            self.set_current_thread(current_thread.id)

        while current_thread.iterations < self.max_iterations:
            print("*" * 100)
            print(f"Agent {self.get_agent_name()} (Thread: {current_thread.id} - Iter: {current_thread.iterations} / {self.max_iterations})")
            print("*" * 100)
            if current_thread.is_started():
                next_state = None
            else:
                next_state = current_thread.get_agent_state()
            print(f"Invoking graph with state: {next_state}")
            self.response = self.agent.graph.invoke(next_state, current_thread.config)
            current_thread.inc_iterations()
            self.partial_message += str(self.response)
            self.partial_message += f"\n------------------\n\n"
            lnode, nnode, _, rev, acount = self.get_display_state()
            
            
            # print(f"After yield")
            config = None # None necessary
            if not nnode:  
                print("Hit the end")
                yield RunnerState(type=RunnerStateType.END)
            elif lnode in stop_after:
                print(f"stopping due to stop_after {lnode} -> {stop_after}")
                yield RunnerState(type=RunnerStateType.STOP)
            else:
                print(f"Not stopping on lnode {lnode} ")
                print(f"thread_id: {self.thread_id_counter},last_node: {lnode}, next_node: {nnode}, rev: {rev}, acount: {acount}")
                yield RunnerState(type=RunnerStateType.ITER,
                                thread_id=self.thread_id_counter, 
                                last_node=lnode, 
                                next_node=nnode if len(nnode) != 0 else None, 
                                message=self.partial_message, 
                                rev=rev, 
                                acount=acount
                )
        yield RunnerState(type=RunnerStateType.MAX_ITERS)
    
    def __repr__(self):
        return f"AgentRunner(agent={self.agent}, max_iterations={self.max_iterations}, share={self.share})"

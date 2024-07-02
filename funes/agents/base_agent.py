from typing import Optional, overload
from langgraph.graph.graph import CompiledGraph

class BaseAgent:

    def __init__(self, name, graph: Optional[CompiledGraph] = None):
        self.name = name
        self.graph = graph
        # Explicit graph injection allows for testing agents with specific graphs        

    def __call__(self):
        if not self.graph:
            self._create_graph()

    @overload
    def _create_graph(self):
        raise NotImplementedError("Subclasses must implement _define_graph() method")


    def get_graph(self):
        if not self.graph:
            self._create_graph()
        return self.graph.get_graph()
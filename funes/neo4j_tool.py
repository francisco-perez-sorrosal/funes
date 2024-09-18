import arxiv
import nest_asyncio
nest_asyncio.apply()

from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright
from playwright_stealth import stealth_sync, stealth_async
from bs4 import BeautifulSoup

from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from typing import Annotated, Literal, Optional, Type

from funes.io_types import BasePaper

# Langchain tool call
# https://github.com/microsoft/autogen/blob/main/notebook/agentchat_langchain.ipynb



from abc import ABC
from typing import Optional
from langchain_community.graphs.neo4j_graph import Neo4jGraph


class GraphMetadata(ABC):
    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
        self.schema = self.graph.get_structured_schema

    def get_node_names(self):
        nodes = self.schema["node_props"]
        return list(nodes.keys())

    def get_edge_names(self, origin: Optional[str] = None, dest: Optional[str] = None):
        
        def filter_edges_on_prop(edges, prop_id, value):
            return [edge for edge in edges if edge[prop_id] == value]
        
        nodes = self.schema["relationships"]
        
        if origin is not None:
            nodes = filter_edges_on_prop(nodes, "start", origin)

        if dest is not None:
            nodes = filter_edges_on_prop(nodes, "end", dest)
        
        return [node['type'] for node in nodes], nodes

    def is_node_in_graph(self, node: str):
        nodes = self.get_node_names()
        return node in nodes

    def get_node_attributes_from_node(self, node):
        attributes = []
        if self.is_node_in_graph(node):
            attributes = [property_info["property"] for property_info in self.schema["node_props"][node]]
        return attributes

    def get_node_instance(self, node: str, instance_id: str, instance_name: str):

        res = ""        
        if self.is_node_in_graph(node):
            query_node_id = f"{node.lower()}"
            res = self.graph.query(f"""
                          MATCH ({query_node_id}:{node}) 
                          WHERE {query_node_id}.{instance_id} = '{instance_name}' 
                          RETURN {query_node_id}
                          """)
        return res


class SmalltalkInput(BaseModel):
    query: Optional[str] = Field(description="user query")

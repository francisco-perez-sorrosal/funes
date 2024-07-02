import arxiv
import json

from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.pydantic_v1 import BaseModel, Field

from typing import Annotated, Literal, Optional, Type


class SearchInput(BaseModel):
    query: str = Field(description="a search query")


class ArxivSearchTool(BaseTool):
    name = "arxiv_search"
    description = "useful for when you need to retrieve info about papers in arxiv"
    args_schema: Type[BaseModel] = SearchInput

    def _run(
        self, query: str, #run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the arxiv search tool."""
        
        client = arxiv.Client()
        search = arxiv.Search(
            query = query,
            max_results = 10,
            sort_by = arxiv.SortCriterion.SubmittedDate
        )

        arxiv_docs = client.results(search)
        
        arxiv_documents = []
        for doc in arxiv_docs:
            arxiv_documents.append({'title': doc.title})
        if len(arxiv_documents) == 0:
            return ""
        json_output = json.dumps({'docs': arxiv_documents})
        return json_output


    async def _arun(
        self, query: str, #run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

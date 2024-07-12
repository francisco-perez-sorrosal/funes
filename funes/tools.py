import arxiv
import json

from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from typing import Annotated, Literal, Optional, Type

from funes.io_types import BasePaper


class SmalltalkInput(BaseModel):
    query: Optional[str] = Field(description="user query")


class SmalltalkTool(BaseTool):
    name = "smalltalk"
    description = "when user greets you or wants to smalltalk"
    args_schema: Type[BaseModel] = SmalltalkInput

    def _run(
        self,
        query: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return (
            "Create a final answer that says if they "
            "have any other question about research papers"
        )



class SearchInput(BaseModel):
    query: str = Field(description="a search query")

class ArxivSearchTool(BaseTool):
    
    name = "arxiv_search"
    description: str = (
        "A wrapper useful to find reliable information about a paper in arxiv. "
        "Input should be a search query."
    )
    args_schema: Type[BaseModel] = SearchInput
    return_direct: bool = True

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> list[BasePaper]:
        """Use the arxiv search tool."""
        
        client = arxiv.Client()
        search = arxiv.Search(
            query = query,
            max_results = 10,
            sort_by = arxiv.SortCriterion.SubmittedDate
        )

        arxiv_docs = client.results(search)
        print(type(arxiv_docs))
        
        arxiv_documents = []
        for doc in arxiv_docs:
            print(doc)
            arxiv_documents.append(BasePaper(source_id=doc.entry_id, title=doc.title, url=doc.pdf_url).json())
        return arxiv_documents
        # json_output = json.dumps(arxiv_documents)
        # return json_output


    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


class ArxivDownloaderTool(BaseTool):
    name = "arxiv_downloader"
    description:str = (
        "A wrapper useful to download a paper from arxiv. "
        "Usually run after the information for a paper has been retrieved. "
        "Input should be a paper information adhering to BasePaper class."
    )
    args_schema: Type[BaseModel] = BasePaper

    def _run(
        self, paper_info: BasePaper, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the arxiv download tool."""
        client = arxiv.Client()
        paper_id = paper_info.source_id
        paper = next(client.results(arxiv.Search(id_list=[paper_id])))
        # Download the PDF to a specified directory with a custom filename.
        paper.download_pdf(dirpath="~/Downloads", filename="funes-test-react.pdf")
        return "Paper downloaded successfully"


    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

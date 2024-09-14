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


class WebScraper:
    def __init__(self, headless=True, browser_type="chromium", chunk_size=256, max_tokens=1000):
        self.headless = headless
        self.browser_type = browser_type
        self.chunk_size = chunk_size
        self.max_tokens = max_tokens

    def scrape_page(self, url: str) -> str:
        with sync_playwright() as p:
            browser = getattr(p, self.browser_type).launch(
                headless=self.headless,
                args=["--disable-gpu", "--no-sandbox"]
            )
            context = browser.new_context()
            page = context.new_page()

            stealth_sync(page)
            page.goto(url)

            html_content = page.content()
            browser.close()
        return html_content


    async def a_scrape_page(self, url: str) -> str:
        # with sync_playwright() as p:
        async with async_playwright() as p:
            browser = await getattr(p, self.browser_type).launch(
                headless=self.headless,
                args=["--disable-gpu", "--no-sandbox"]
            )
            context = await browser.new_context()
            page = await context.new_page()
            await stealth_async(page)
            await page.goto(url)

            html_content = await page.content()
            await browser.close()
        return html_content

    def extract_titles_articles_links(self, raw_html: str) -> list:
        soup = BeautifulSoup(raw_html, 'html.parser')
        extracted_data = []
        visited_links = set()
        for article in soup.find_all(['article', 'section', 'div']):
            title_tag = article.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            link_tag = article.find('a', href=True)
            content = article.get_text(separator="\n", strip=True)
            
            if title_tag and link_tag and content and link_tag['href'] not in visited_links:
                extracted_data.append({
                    'title': title_tag.get_text(strip=True),
                    'link': link_tag['href'],
                    'content': content
                })
                visited_links.add(link_tag['href'])
        
        return extracted_data

    def query_page_content(self, url: str) -> dict:
        raw_html = self.scrape_page(url)
        structured_data = {
            "url": url,
            "extracted_data": self.extract_titles_articles_links(raw_html),
            "raw_html": raw_html
        }
        return structured_data


    async def a_query_page_content(self, url: str) -> dict:
        raw_html = await self.a_scrape_page(url)
        structured_data = {
            "url": url,
            "extracted_data": self.extract_titles_articles_links(raw_html),
            "raw_html": raw_html
        }
        return structured_data



def query_web_scraper(url: str) -> dict:
    scraper = WebScraper(headless=True)
    return scraper.query_page_content(url)


async def a_query_web_scraper(url: str) -> dict:
    scraper = WebScraper(headless=True)
    return await scraper.a_query_page_content(url)

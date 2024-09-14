import asyncio
import os
import re

from typing import List, Type

import matplotlib.pyplot as plt
import pprint
import streamlit as st

from autogen.agentchat import GroupChat, GroupChatManager
from autogen.graph_utils import visualize_speaker_transitions_dict

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools import TavilySearchResults
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool

from llm_foundation import logger
from funes.agents.agent_types import AutogenAgentType, Persona, always_terminate
from funes.tools import a_query_web_scraper, query_web_scraper


# os.environ["TAVILY_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Funes"

st.title("Bibi, The BibTex Agent")

# Get config from session
llm_config = st.session_state.get('lm_config', None)
if llm_config is None:
    st.write("No LLM config found")
    st.stop()


st.sidebar.header(f"LLM Config")
st.sidebar.json(llm_config)


class Document(BaseModel):
    title: str = Field("The title of the document representing the search result", example="Capital of France - Wikipedia")
    raw_content: str = Field("The snippet of the search result", example="Paris is the capital of France and...")

class WebDocument(Document):
    uri: str = Field("The link to the document", example="https://en.wikipedia.org/wiki/Paris")
    is_snippet: bool = Field("Whether the document is a snippet or not", example=False)
    
class SearchDocuments(BaseModel):
    query: str = Field("The query that lead to the docs obtained", example="What is the capital of France?")
    docs: List[WebDocument] = []



def transform_ddg_results_to_search_documents(query, content):
    pattern = r"\[snippet: (.*?), title: (.*?), link: (.*?)\]"
    matches = re.findall(pattern, content)
    
    search_docs = SearchDocuments(query=query)
    
    for match in matches:
        search_result = WebDocument(raw_content=match[0], title=match[1], uri=match[2], is_snippet=True)
        search_docs.docs.append(search_result)
    return search_docs

def transform_tavily_results_to_search_documents(query, tavily_results: List[dict]):
    search_docs = SearchDocuments(query=query)
    for doc in tavily_results:
        search_result = WebDocument(raw_content=doc['content'], title="", uri=doc['url'], is_snippet=False)
        search_docs.docs.append(search_result)
    return search_docs


class SearchToolInput(BaseModel):
    query: str = Field("The query to search for", example="What is the capital of France?")
    use_smart_search: bool = Field("Whether to use smart search or not", example=True)
    num_results: int = Field("The number of results to return", example=5)


class FunesSearch(BaseTool):
    name = "funes_search"
    description = "Use this tool when you need to search for documents related to a query in the web."
    args_schema: Type[BaseModel] = SearchToolInput

    def _run(self, query: str, use_smart_search: bool):
        logger.info(f"Funes Search for {query}\nuse_smart_search={use_smart_search}")
        if use_smart_search:
            tool = TavilySearchResults(
                max_results=5,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=False,
                include_images=False,
                # include_domains=[...],
                # exclude_domains=[...],
                # name="...",            # overwrite default tool name
                # description="...",     # overwrite default tool description
                # args_schema=...,       # overwrite default args_schema: BaseModel
            )
            results = tool.invoke({"query":query})
            logger.info(f"Search result ({type(results)}):\n{pprint.pprint(results)}")
            transformed_result = transform_tavily_results_to_search_documents(query, results)
        else:
            def ddg_search(query: str):
                """Search the web for information about a particular query"""
                logger.info("Modifying the query to search in google books")
                query = query + " google books"
                return DuckDuckGoSearchResults(verbose=True, response_format='content', num_results=5).run(query, verbose=True, return_json=True)

            results = ddg_search(query)
            logger.info(f"Search result ({type(results)}):\n{results}")
            transformed_result = transform_ddg_results_to_search_documents(query, results)
        
        return transformed_result

    
class WebCrawlerToolInput(BaseModel):
    query: str = Field("The query that lead to the docs obtained", example="What is the capital of France?")
    docs: SearchDocuments = Field("The text where to extract its title, content or a uri/url link from")


class WebCrawler(BaseTool):
    name = "web_crawler"
    description = "Use this tool when you have to extract content from a web page."
    args_schema: Type[BaseModel] = WebCrawlerToolInput

    def _run(self, input_docs: SearchDocuments):
    
        input_docs = SearchDocuments.parse_obj(input_docs)  # Convert to pydantic object
                
        blob = ""
        for search_result in input_docs.docs:
            url = search_result.uri
            logger.info(f"Extracting data from url {url}")
            extracted_metadata = query_web_scraper(url)

            content = ""
            for content_chunk in extracted_metadata["extracted_data"]:
                logger.info(f"Chunk size: {len(content_chunk['content'])}")
                if len(content_chunk["content"]) < 100:
                    logger.info(f"Skipping content chunk: {content_chunk}")
                    continue
                content = content + " " + content_chunk["title"] + " " + content_chunk["content"][:100]            
            blob = blob + content

        logger.info(f"Extracted blob: {blob}")
        return blob
    

    async def _arun(self, search_results: SearchDocuments):
        # TODO Not important for now
        url = "https://medium.com/javarevisited/what-i-learned-from-the-book-system-design-interview-an-insider-guide-77562e48cdaa"
        html = asyncio.create_task(a_query_web_scraper(url))
        # loop = asyncio.get_event_loop()
        # html = loop.run_until_complete(a_query_web_scraper(url))
        await html
        logger.info(f"Extracting link async from {search_results.docs[0].uri}")
        return "KK"

        
agent_tab, = st.tabs(["Agent"])

messages = []
with agent_tab:

    # with st.container():
        
    francisco = Persona.from_json_file("notebooks/Persona/Francisco.json")        
    researcher = Persona.from_json_file("notebooks/Persona/Researcher.json")
    maestro = Persona.from_json_file("notebooks/Persona/Maestro.json")
    
    st.sidebar.header("Agent Roles")
    st.sidebar.markdown("## Users")
    st.sidebar.write(francisco)
    st.sidebar.markdown("## Researchers")        
    st.sidebar.write(researcher)
    st.sidebar.markdown("## Maestros")
    st.sidebar.write(maestro)


    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "selected_question" not in st.session_state:
        st.session_state.selected_question = []


    queries = [
        "System Design Interview – An insider's guide",
        "System Design Interview Volume 2",
        "Machine Learning Chip Huyen",
    ]
    
    selected_query = st.selectbox("Select an example book...", queries)
    prompt = st.text_input("... or type a book title (optionally with author/s)...", key="book_title", value=selected_query)
    search = st.button("Search")        
        
    with st.spinner(f"Processing query: {prompt}"):    
        if search:
        
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            ddg_search_tool = FunesSearch()
            ddg_search_oai_tool = convert_to_openai_function(ddg_search_tool)
            
            web_crawler_tool = WebCrawler()
            web_crawler_oai_tool = convert_to_openai_function(web_crawler_tool)
                        
            tools = [ddg_search_oai_tool, web_crawler_oai_tool]
            
            llm_with_tools_config = llm_config.copy()
            llm_with_tools_config.update({"functions": tools})
            
            llm_with_json_responses_config = llm_config.copy()
            llm_with_json_responses_config.update({"response_format": { "type": "json_object" }})
            
            book_web_finder = researcher.role_to_autogen_agent("book_finder", AutogenAgentType.AssistantAgent, llm_config=llm_with_tools_config)
            book_web_finder.register_function(
                function_map={
                    ddg_search_tool.name: ddg_search_tool._run
                }
            )
            web_crawler = researcher.role_to_autogen_agent("web_crawler", AutogenAgentType.AssistantAgent, llm_config=llm_with_tools_config)
            web_crawler.register_function(
                function_map={
                    web_crawler_tool.name: web_crawler_tool._run
                }
            )            

            doc_info_json_extractor = researcher.role_to_autogen_agent("json_info_extractor", AutogenAgentType.AssistantAgent, llm_config=llm_with_json_responses_config)                        
            bibtex_writer = maestro.role_to_autogen_agent("bib_tex_writer", AutogenAgentType.AssistantAgent, llm_config=llm_config)

            user_proxy = francisco.role_to_autogen_agent("learner", AutogenAgentType.UserProxyAgent, llm_config=llm_config, termination_function=always_terminate)
            
            agent_speaker_transitions_dict = {
                user_proxy: [book_web_finder],
                book_web_finder: [web_crawler, doc_info_json_extractor],
                web_crawler: [doc_info_json_extractor],
                doc_info_json_extractor: [bibtex_writer],
                bibtex_writer: [user_proxy]
            }
            
            agent_crew = [user_proxy, book_web_finder, web_crawler, doc_info_json_extractor, bibtex_writer]

            # Visualize agent transitions in sidebar
            st.sidebar.write("Display agent Crew")
            fig = plt.figure(figsize=(8,8))
            visualize_speaker_transitions_dict(agent_speaker_transitions_dict, agent_crew)
            with st.sidebar:
                st.pyplot(fig)

            # Create group chat and initiate chat
            groupchat = GroupChat(
                agents = agent_crew,
                messages=[],
                max_round=10,
                select_speaker_auto_verbose=True,
                speaker_transitions_type="allowed",  # This has to be specified if the transitions below apply
                allowed_or_disallowed_speaker_transitions=agent_speaker_transitions_dict,
            )
            
            manager = GroupChatManager(
                groupchat=groupchat, 
                llm_config=llm_config,
                system_message="You act as a coordinator for different specialiced roles. If you don't have anything to say, just say TERMINATE."
            )        
            
            response = user_proxy.initiate_chat(
                manager,
                message=prompt,
            )
            
            def find_last_message(name: str, chat_history):
                for message in reversed(chat_history):
                    if message["name"] == name:
                        return message
                return None
            
            last_message = find_last_message("bib_tex_writer", response.chat_history)
            last_message_content = "No final response found from the writer." if last_message is None else last_message["content"]
            st.session_state.messages.append({"role": "assistant", "content": last_message_content})
            
    st.header(f"Queries and Responses")
    for i in range(len(st.session_state.messages) - 1, 0, -2):
        message = st.session_state.messages[i-1]
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        message = st.session_state.messages[i]
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


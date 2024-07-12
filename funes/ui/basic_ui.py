import os
import streamlit as st

from funes.lang_utils import models, get_llm
from funes.agents.basic_agent import BasicAgent, Role, TOOLS
from funes.prompt import basic_react_prompt
from funes.agents.coordinator import ArxivRetrieverAgenticGraph

from IPython.display import Image, display

from funes.utils import print_event


with open("langsmith", "r") as file:
    langsmith_api_key = file.read().strip()
st.write(f"Langsmith API key: {langsmith_api_key}")
os.environ["LANGSMITH_API_KEY"] = langsmith_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Funes"

with open("openai", "r") as file:
    openai_api_key = file.read().strip()
st.write(f"Open API key: {openai_api_key}")
os.environ["OPEN_API_KEY"] = openai_api_key

st.title("Basic UI")

lm = st.session_state.get("lm")

if not lm:
    st.warning("No model selected")
    st.stop()
    

agent = BasicAgent(lm, basic_react_prompt)

if prompt := st.chat_input("How much does a toy poodle weigh?"):
    st.chat_message(Role.USER).write(prompt)
    result=agent.query(prompt, known_tools=TOOLS)
    
    st.write(result)
    
    st.write(agent.messages)

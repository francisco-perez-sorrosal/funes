import os
import streamlit as st

from funes.lang_utils import models, get_llm
from funes.agents.retriever import RetrieverAgent

from IPython.display import Image, display

from funes.utils import print_event


with open("langsmith", "r") as file:
    langsmith_api_key = file.read().strip()
st.write(f"Langsmith API key: {langsmith_api_key}")
os.environ["LANGSMITH_API_KEY"] = langsmith_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Funes"

st.title("Retriever UI")

lm = st.session_state.get("lm")

if not lm:
    st.warning("No model selected")
    st.stop()
   
retriever = RetrieverAgent(lm)
    
st.image(retriever.get_graph().draw_png())

if prompt := st.chat_input('Retrieve the ReAct paper from arxiv 2210.03629'):
    st.chat_message("user").write(prompt)
    for event in retriever(prompt):
        print_event(event, _printed=set(), streamlit=True)     

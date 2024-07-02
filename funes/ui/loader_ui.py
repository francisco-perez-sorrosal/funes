import streamlit as st

import arxiv

from funes.lang_utils import models, get_llm
from funes.agents.retriever import RetrieverAgent

from IPython.display import Image, display

from funes.utils import print_event



# display(Image(graph.get_graph().draw_mermaid_png()))


lm = st.session_state.get("lm")

if not lm:
    st.warning("No model selected")
    st.stop()

# with st.form('retriever'):
    # text = st.text_area('Enter text:', 'Retrieve the ReAct paper from arxiv 2210.03629')
    
    # submitted = st.form_submit_button('Submit')
    
retriever = RetrieverAgent(lm)
    
    # st.write(type(retriever.get_graph()))
    
st.image(retriever.get_graph().draw_png())

if prompt := st.chat_input('Retrieve the ReAct paper from arxiv 2210.03629'):
    st.chat_message("user").write(prompt)
    for event in retriever(prompt):
        print_event(event, _printed=set(), streamlit=True)
        

        

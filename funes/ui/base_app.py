import streamlit as st

from funes.lang_utils import models, get_llm


st.set_page_config(layout="wide")

providers = ["HF", "OpenAI", "OllamaLocal", "TGI"]
model_options = list(models.keys())


def page2():
    st.title("Second page")

pg = st.navigation([
    st.Page("basic_ui.py", title="Basic agent", icon="🔥"),
    st.Page("retriever_ui.py", title="Agent loader", icon="🔥"),
    st.Page(page2, title="Second page", icon=":material/favorite:"),
])

lm = st.session_state.get("lm")
if lm is None:
    st.sidebar.markdown("## Select LLM")
    with st.sidebar.form('select_lm_form'):
        # Dropdown box for selecting provider options
        provider = st.selectbox("Select a provider", providers, index=0)
        model = None
        if provider != "HF":
            selected_model = st.selectbox("Select a model", model_options, index=len(model_options)-1)
            model = models.get(selected_model, "")
        if model == "":
            st.warning("No model selected")
            st.stop() 
        
        select_btn = st.form_submit_button('Select LM')
        if select_btn:
            st.write(f"Model {model}")
            lm = get_llm(model, provider, port=8081)
            st.session_state["lm"] = lm
            st.sidebar.info(f"LM {lm} selected")
            st.rerun()
        else:
            st.sidebar.warning(f"No LM selected")
else:
    st.sidebar.info(f"LM {lm} selected")
    with st.sidebar.form('reset_lm_form'):
        reset_btn = st.form_submit_button('Reset LM')
        if reset_btn:
            st.sidebar.warning("LM reset")
            del st.session_state["lm"]
            st.rerun()

pg.run()

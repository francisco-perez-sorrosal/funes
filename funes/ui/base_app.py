import streamlit as st

# from funes.lang_utils import models, get_llm
from llm_foundation.lm import get_lm, get_model_catalog
from llm_foundation.basic_structs import Provider, LMConfig


st.set_page_config(layout="wide")




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
        providers = [p.name for p in Provider]        
        provider = st.selectbox("Select a provider", providers, index=0)
        provider_as_enum = Provider[provider]
        models = get_model_catalog(provider_as_enum, ["gpt-4o"])
        model_options = [k for k in models.keys() if k.startswith(provider.lower())]
        selected_model = st.selectbox("Select a model", model_options, index=0)
        model = models.get(selected_model, "")
        if model == "":
            st.warning("No model selected")
            st.stop() 
        
        select_btn = st.form_submit_button('Select LM')
        if select_btn:
            st.write(f"Model name {model}")
            lm_config = LMConfig(model=model, provider=provider_as_enum)
            lm = get_lm(lm_config)
            # lm = get_llm(model, provider, port=8081)
            st.session_state["lm"] = lm
            st.sidebar.info(f"LM selected:\n{lm}")
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

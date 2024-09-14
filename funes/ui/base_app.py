import streamlit as st

# from funes.lang_utils import models, get_llm
from llm_foundation.lm import get_lm, get_model_catalog
from llm_foundation.basic_structs import Provider, LMConfig


st.set_page_config(layout="wide")

def page2():
    st.title("Second page")

pg = st.navigation([
    st.Page("bibtex_ui.py", title="Bibi the agent!", icon="🔥"),
    st.Page("learner_agent_ui.py", title="Learner agent", icon="🔥"),
])

lm = st.session_state.get("lm")
if lm is None:
    st.sidebar.markdown("## Select LLM")
    with st.sidebar.form('select_lm_form'):
        # Dropdown box for selecting provider options
        providers = [p.name for p in Provider]        
        provider = st.selectbox("Select a provider", providers, index=2)
        provider_as_enum = Provider[provider]
        models = get_model_catalog(provider_as_enum, ["gpt-4o-mini"])
        model_options = [k for k in models.keys() if k.startswith(provider.lower())]
        selected_model = st.selectbox("Select a model", model_options, index=0)
        # selected_model = st.selectbox("Select a model", ['gpt-4'], index=0)
        model = models.get(selected_model, "")
        select_btn = st.form_submit_button('Select LM')
        if model == "":
            st.warning("No model selected")
            st.stop() 
        
        if select_btn:
            lm_config = LMConfig(model=model, provider=provider_as_enum)
            st.session_state["lm_config"] = lm_config.to_autogen()
            st.session_state["lm"] = get_lm(lm_config)
            st.sidebar.info(f"LM selected:\n{st.session_state['lm']}")
            st.rerun()
        # else:
        #     st.sidebar.warning(f"No LM selected")
else:
    st.sidebar.info(f"LM {lm} selected")
    with st.sidebar.form('reset_lm_form'):
        reset_btn = st.form_submit_button('Reset LM')
        if reset_btn:
            st.sidebar.warning("LM reset")
            del st.session_state["lm"]
            st.rerun()

pg.run()

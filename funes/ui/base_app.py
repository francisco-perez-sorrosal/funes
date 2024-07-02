
import importlib.util
import multiprocessing
import os
import pathlib

from typing import Optional, Sequence

import dspy
import streamlit as st
import pandas as pd

from dspy.datasets.hotpotqa import HotPotQA
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dsp.utils import flatten, deduplicate
from funes.interactions import FactualSortQA
from funes.lang_utils import models, get_llm





st.set_page_config(layout="wide")

providers = ["HF", "OA", "OllamaLocal", "TGI"]
model_options = list(models.keys())


def page2():
    st.title("Second page")

pg = st.navigation([
    st.Page("loader_ui.py", title="Agent loader", icon="🔥"),
    st.Page(page2, title="Second page", icon=":material/favorite:"),
])

lm = st.session_state.get("lm")
if lm is None:
    st.sidebar.markdown("## Select LLM")
    with st.sidebar.form('select_lm_form'):
        # Dropdown box for selecting options
        provider_option = st.selectbox("Select a provider", providers, index=0)
        provider = provider_option
        model = None
        if provider != "HF":
            selected_option = st.selectbox("Select an option", model_options, index=len(model_options)-1)
            model = models.get(selected_option, "")
        if model == "":
            st.warning("No model selected")
            st.stop() 
        
        select_btn = st.form_submit_button('Select LM')
        if select_btn:
            st.write(f"Model {model}")
            lm = get_llm(model, provider, port=8081)
            st.session_state["lm"] = lm
            st.sidebar.info(f"LM {lm} selected")
            st.experimental_rerun()
        else:
            st.sidebar.warning(f"No LM selected")
    
else:
    
    st.sidebar.info(f"LM {lm} selected")
    with st.sidebar.form('reset_lm_form'):
        reset_btn = st.form_submit_button('Reset LM')
        if reset_btn:
            st.sidebar.warning("LM reset")
            del st.session_state["lm"]
            st.experimental_rerun()

pg.run()

import dspy
import streamlit as st

models = {
    "Llama3-ChatQA-1.5-8B": "nvidia/Llama3-ChatQA-1.5-8B",
    "gemma-7b": "google/gemma-7b",
    "Phi-3-small-8k-instruct": "microsoft/Phi-3-small-8k-instruct",
    "Llama-2-7b-hf": "meta-llama/Llama-2-7b-hf",
    "llama3": "llama3",
    "hf-llama3": "meta-llama/Meta-Llama-3-8B",
    "hf-llama3-inst": "meta-llama/Meta-Llama-3-8B-Instruct",
    "hf-phi3-mini-4k-inst": "microsoft/Phi-3-mini-4k-instruct",
    "hf-phi3-small-4k-inst": "microsoft/Phi-3-small-8k-instruct",
    "hf-gemma-7b-it": "google/gemma-7b-it",
}

@st.cache_resource(show_spinner=True)
def get_llm(model: str, provider: str, port: int=8081, url: str="http://localhost", temperature: float=0.0, max_tokens: int=100):
    if provider == "OllamaLocal":
        lm = dspy.OllamaLocal(model=model, port=port, temperature=temperature, max_tokens=max_tokens)
    elif provider == "VLLM":
        lm = dspy.HFClientVLLM(model=model, port=port, url=url, temperature=temperature, max_tokens=max_tokens)
    elif provider == "TGI":
        lm = dspy.HFClientTGI(model=model, port=port, url=url, temperature=temperature, max_tokens=max_tokens)
    else:
        raise ValueError("Invalid backend provider")
    
    return lm

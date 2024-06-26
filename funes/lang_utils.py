import os

import dspy
import streamlit as st

models = {
    "oai-gpt3.5-turbo": "gpt-3.5-turbo-1106",
    "oai-gpt4-turbo": "gpt-4-turbo",
    "hf-nvidia-Llama3-ChatQA-1.5-8B": "nvidia/Llama3-ChatQA-1.5-8B",
    "hf-Llama-2-7b-hf": "meta-llama/Llama-2-7b-hf",
    "hf-Meta-Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "hf-Phi-3-mini-4k-instruct": "microsoft/Phi-3-mini-4k-instruct",
    "hf-Phi-3-small-8k-instruct": "microsoft/Phi-3-small-8k-instruct",
    "hf-gemma-7b-it": "google/gemma-7b-it",
    "hf-Meta-Llama-3-8B": "meta-llama/Meta-Llama-3-8B",
    
    "hf-Meta-Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
}

OA_API_KEY = os.getenv("OA_API_KEY")

@st.cache_resource(show_spinner=True)
def get_llm(model: str, provider: str, port: int=8081, url: str="http://localhost", temperature: float=0.0, max_tokens: int=100):
    if provider == "OA":
        lm = dspy.OpenAI(model=model, api_key=OA_API_KEY, max_tokens=max_tokens)
    elif provider == "OllamaLocal":
        lm = dspy.OllamaLocal(model=model, port=port, temperature=temperature, max_tokens=max_tokens)
    elif provider == "VLLM":
        lm = dspy.HFClientVLLM(model=model, port=port, url=url, temperature=temperature, max_tokens=max_tokens)
    elif provider == "TGI":
        lm = dspy.HFClientTGI(model=model, port=port, url=url, temperature=temperature, max_tokens=max_tokens)
    else:
        raise ValueError("Invalid backend provider")
    
    return lm

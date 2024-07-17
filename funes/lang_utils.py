import os
import openai
import langchain_openai

from typing import Optional

import dspy
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


providers = ["HF", "OpenAI-Chat", "OpenAI", "OllamaLocal", "TGI"]

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

with open("openai", "r") as file:
    openai_api_key = file.read().strip()
os.environ["OPENAI_API_KEY"] = openai_api_key

@st.cache_resource(show_spinner=True)
def get_llm(model: Optional[str], provider: str="HF", port: int=8081, url: str="http://localhost", temperature: float=0.0, max_tokens: int=100):
    match provider:
        # case "HF-Chat":
        #     lm = ChatOpenAI(
        #         model_name="tgi",                
        #         openai_api_base=f"{url}:{port}" + "/v1/",
        #     )
        case "HF":
            callbacks = [StreamingStdOutCallbackHandler()]
            lm = HuggingFaceEndpoint(
                endpoint_url=f"{url}:{port}",
                task="text-generation",
                max_new_tokens=max_tokens,
                top_k=10,
                top_p=0.95,
                typical_p=0.95,
                temperature=0.01,
                repetition_penalty=1.03,
                callbacks=callbacks,
                streaming=True,
                do_sample=False,
                stop_sequences=["<|eot_id|>"]
            )
        case "OpenAI-Chat":
            lm = langchain_openai.ChatOpenAI(model="gpt-4o")                        
        case "OpenAI":
            lm = dspy.OpenAI(model=model, api_key=os.environ.get("OPENAI_API_KEY"), max_tokens=max_tokens)
            lm = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        case "OllamaLocal":
            lm = dspy.OllamaLocal(model=model, port=port, temperature=temperature, max_tokens=max_tokens)
        case "VLLM":
            lm = dspy.HFClientVLLM(model=model, port=port, url=url, temperature=temperature, max_tokens=max_tokens)
        case "TGI":
            lm = dspy.HFClientTGI(model=model, port=port, url=url, temperature=temperature, max_tokens=max_tokens)
        case _:
            raise ValueError("Invalid backend provider")
    
    return lm

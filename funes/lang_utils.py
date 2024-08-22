# To Remove in favor of functionality on llm_foundation
git 
import os
import openai
import langchain_openai

from typing import Optional

import dspy
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI, OpenAI
from langchain_ollama import ChatOllama
# from langchain_experimental.llms.ollama_functions import OllamaFunctions  # Extends of ChatOllama
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


providers = ["Ollama", "OpenAI", "HF"]

models = {
    "openai-gpt4o-latest": "gpt-4o-2024-08-06",
    "openai-gpt3.5-turbo": "gpt-3.5-turbo-1106",
    "openai-gpt4-turbo": "gpt-4-turbo",
    "hf-nvidia-Llama3-ChatQA-1.5-8B": "nvidia/Llama3-ChatQA-1.5-8B",
    "hf-Llama-2-7b-hf": "meta-llama/Llama-2-7b-hf",
    "hf-Meta-Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "hf-Phi-3-mini-4k-instruct": "microsoft/Phi-3-mini-4k-instruct",
    "hf-Phi-3-small-8k-instruct": "microsoft/Phi-3-small-8k-instruct",
    "hf-gemma-7b-it": "google/gemma-7b-it",
    "hf-Meta-Llama-3-8B": "meta-llama/Meta-Llama-3-8B",
    "hf-Meta-Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "ollama-llama3.1": "llama3.1",
    "ollama-llama3-groq-tool-use": "llama3-groq-tool-use",
}

with open("openai", "r") as file:
    openai_api_key = file.read().strip()
os.environ["OPENAI_API_KEY"] = openai_api_key

from langchain_core.runnables.utils import ConfigurableField

@st.cache_resource(show_spinner=True)
def get_llm(model: Optional[str], provider: str="OpenAI", want_chat: bool=True, port: int=8081, url: str="http://localhost", temperature: float=0.0, max_tokens: int=1000):
    match provider:
        case "Ollama":
            if want_chat:
                import instructor

                openai_chat = ChatOpenAI(
                    base_url="http://localhost:11434/v1",
                    api_key="ollama",  # required, but unused
                )
                chat_completions = openai_chat.client
                updated_openai_client = instructor.from_openai(chat_completions._client)
                chat_completions._client = updated_openai_client
                openai_chat = openai_chat.copy(update={'client': chat_completions})
                
                lm = ChatOllama(
                    model=model,
                    temperature=0.0,
                    # format='json',
                    max_tokens=1000,
                ).configurable_alternatives(
                    ConfigurableField(id="llm"),
                    default_key="ollama",
                    openai=openai_chat
                )
        case "OpenAI":
            if want_chat:
                st.write(f"Model {model}")
                print(f"Model {model}")
                lm = ChatOpenAI(
                    model=model,
                    temperature=0.0,
                )
                print(lm)
        # case "HF":
        #     if want_chat:
        #     lm = ChatOpenAI(
        #         model_name="tgi",                
        #         openai_api_base=f"{url}:{port}" + "/v1/",
        #     )
        case _:
            raise ValueError("Invalid backend provider")
    
    return lm

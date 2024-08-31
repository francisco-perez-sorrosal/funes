import os
from collections.abc import Generator
from typing import Optional
import streamlit as st
import asyncio

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage

from langgraph.checkpoint.sqlite import SqliteSaver

from IPython.display import Image, display

from funes.agents.agent_types import AutogenAgentType, Persona
from funes.agents.maestro_app import reflection_message


with open("langsmith", "r") as file:
    langsmith_api_key = file.read().strip()
# st.write(f"Langsmith API key: {langsmith_api_key}")
os.environ["LANGSMITH_API_KEY"] = langsmith_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Funes"

with open("openai", "r") as file:
    openai_api_key = file.read().strip()
# st.write(f"Open API key: {openai_api_key}")
os.environ["OPEN_API_KEY"] = openai_api_key


st.title("Agent")

lm = st.session_state.get("lm")
if not lm:
    st.warning("No model selected")
    st.stop()


if not openai_api_key or not lm:
    st.warning(
        'You must provide valid OpenAI API key and choose preferred model', icon="⚠️")
    st.stop()

config_list = [{
    "model": lm,  # model name
    "api_key": openai_api_key  # api key
}]
llm_config = {
    "seed": 14,  # seed for caching and reproducibility
    "config_list": config_list,  # a list of OpenAI API configurations
    "temperature": 0.0,  # temperature for sampling
}


st.sidebar.header("Agent Roles")


### Status bars
DEFAULT_STATUS = "Ready"

agent_col, status_col = st.columns([1,1])
with agent_col:
    agent_status = st.empty()


with status_col:
    status_bar = st.empty()



        
agent_tab, plan_tab, snapshot_tab = st.tabs(["Agent", "Plan", "Snapshot"])

messages = []
with agent_tab:

    with st.container():
        # for message in st.session_state["messages"]:
        #    st.markdown(message)
        
        francisco = Persona.from_json_file("notebooks/Persona/Francisco.json")        
        maestro = Persona.from_json_file("notebooks/Persona/Maestro.json")
        critics = Persona.from_json_file("notebooks/Persona/Critic.json")

        st.sidebar.write(francisco)
        st.sidebar.write(maestro)
        st.sidebar.write(critics)


        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "selected_question" not in st.session_state:
            st.session_state.selected_question = []




        questions = [
            "What is cross-validation and why is it used in machine learning?",
            "What is cross-validation and why is it used in machine learningsdsdsdd?",
        ]
        
        def update_selected_question():
            st.write(st.session_state.sb_question)
            st.session_state.selected_question.append(st.session_state.sb_question)
            st.write(len(st.session_state.selected_question))

        selected_question = st.selectbox("Select a question", questions, key="sb_question", on_change=update_selected_question)
        

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])        
        
        if prompt:= st.chat_input("Type something..."):
            
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
                        
            maestro_teacher = maestro.role_to_autogen_agent("generic_teacher", AutogenAgentType.AssistantAgent, llm_config=llm_config)

            science_critic = critics.role_to_autogen_agent("science_critic", AutogenAgentType.AssistantAgent, llm_config=llm_config)
            style_editor = critics.role_to_autogen_agent("style_editor", AutogenAgentType.AssistantAgent, llm_config=llm_config)
            ethics_reviewer = critics.role_to_autogen_agent("ethics_reviewer", AutogenAgentType.AssistantAgent, llm_config=llm_config)
            meta_reviewer = critics.role_to_autogen_agent("meta_reviewer", AutogenAgentType.AssistantAgent, llm_config=llm_config)

            review_chats = [
                {
                    "recipient": ethics_reviewer, 
                    "message": reflection_message, 
                    "summary_method": "reflection_with_llm",
                    "summary_args": {"summary_prompt" : 
                    "Return review into as JSON object only:"
                    "{'Reviewer': '', 'Review': ''}. Here Reviewer should be your role",},
                    "max_turns": 1},
                {
                    "recipient": style_editor, 
                    "message": reflection_message, 
                    "summary_method": "reflection_with_llm",
                    "summary_args": {"summary_prompt" : 
                    "Return review into as JSON object only:"
                    "{'Reviewer': '', 'Review': ''}. Here Reviewer should be your role",},
                    "max_turns": 1},    
                {
                    "recipient": meta_reviewer, 
                    "message": "Aggregrate feedback from all reviewers and give final suggestions on the writing.",
                    "max_turns": 1},
            ]

            science_critic.register_nested_chats(
                review_chats,
                trigger=maestro_teacher,
            )
            
            
            response = science_critic.initiate_chat(
                recipient=maestro_teacher,
                message=prompt,
                max_turns=2,
                summary_method="last_msg"
            )
            
            
            with st.chat_message("assistant"):
                response = st.markdown(response.summary)
            # Display assistant response in chat message container
            st.session_state.messages.append({"role": "assistant", "content": response.summary})


        default_chat_input_value = "Default Value"
        js = f"""
            <script>
                function insertText(dummy_var_to_force_repeat_execution) {{
                    var chatInput = parent.document.querySelector('textarea[data-testid="stChatInputTextArea"]');
                    var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
                    nativeInputValueSetter.call(chatInput, "{default_chat_input_value}");
                    var event = new Event('input', {{ bubbles: true}});
                    chatInput.dispatchEvent(event);
                }}
                insertText({len(st.session_state.messages)}));
            </script>
            """
        st.components.v1.html(js)




    col1, col2 = st.columns([1,1])
    with col1:
        st.write("Col1")
    with col2:
        st.write("Col2")

with plan_tab:
    st.write("Plan Tab")

with snapshot_tab:
    st.header("Snapshots Tab")

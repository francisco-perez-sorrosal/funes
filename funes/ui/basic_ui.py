import os
import streamlit as st

from funes.lang_utils import models, get_llm
from funes.agents.basic_agent import BasicAgent, BasicLGAgent, Role, TOOLS, langchain_tools
from funes.prompt import basic_react_prompt, basic_template
from funes.agents.coordinator import ArxivRetrieverAgenticGraph
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from funes.ui.agent_state import show_agent_state, show_agent_messages
# from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
from langgraph.checkpoint.sqlite import SqliteSaver

from IPython.display import Image, display

from funes.utils import print_event


with open("langsmith", "r") as file:
    langsmith_api_key = file.read().strip()
st.write(f"Langsmith API key: {langsmith_api_key}")
os.environ["LANGSMITH_API_KEY"] = langsmith_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Funes"

with open("openai", "r") as file:
    openai_api_key = file.read().strip()
st.write(f"Open API key: {openai_api_key}")
os.environ["OPEN_API_KEY"] = openai_api_key

st.title("Basic UI")

lm = st.session_state.get("lm")

if not lm:
    st.warning("No model selected")
    st.stop()
    

memory = st.session_state.get("memory")

if not memory:
    memory = SqliteSaver.from_conn_string(":memory:")
    st.session_state["memory"] = memory
    st.warning(f"Memory created and added to the session state: {memory}")


thread = st.session_state.get("thread")
if not thread:
    st.warning("No thread selected")
    thread = {"configurable": {"thread_id": "1"}}
    st.session_state["thread"] = thread
    st.info(f"Thread {thread} created and added to the session state")


agent = st.session_state.get("agent")
if not agent:
    agent_type = st.selectbox("Agent type", ["Basic", "LG-Based"], index=1)

    if agent_type == "Basic":
        agent = BasicAgent(lm, basic_react_prompt)
        st.session_state["agent_type"] = "Basic"
    else:
        agent = BasicLGAgent(lm, langchain_tools, basic_template, memory)
        st.session_state["agent_type"] = "LG-Based"
    st.session_state["agent"] = agent
    st.info(f"Agent {agent.name} created and added to the session state")
    

st.write(f"Agent: {agent.name}, Thread: {thread}")


messages = []
if user_prompt := st.chat_input("How much does a toy poodle weight?"):
    st.chat_message(Role.USER).write(user_prompt)
 
    agent_type = st.session_state.get("agent_type")   
    if agent_type == "Basic":
        result=agent.query(user_prompt, known_tools=TOOLS)
    else:
        messages = [HumanMessage(content=user_prompt)]

if user_prompt:

    # result=agent.graph.invoke({"messages": messages})  # Simple
    for event in agent.graph.stream({"messages": messages}, thread):
        for evnt in event.values():
            show_agent_messages(evnt)
else:
    if agent:
        st.write(f"### Last query events for thread {thread['configurable']['thread_id']}")
        try:
            for event in agent.graph.stream(None, thread):
                for evnt in event.values():
                    # st.write(evnt)
                    # st.write(type(evnt))
                    show_agent_messages(evnt["messages"])
        except ValueError as e:
            st.warning(e)
            


if agent:
    while agent.graph.get_state(thread).next:
        
        agent_state = agent.graph.get_state(thread)
        # st.write(type(agent_state))
        show_agent_state(agent_state)
        # st.chat_message(role).write(agent_state.values)
        proceed_btn = st.button(f"proceed? ({agent_state.created_at})", key=agent_state.created_at)
        if proceed_btn:
            st.write("Proceed pressed")
            for event in agent.graph.stream(None, thread):
                for v in event.values():
                    st.write(v)
        else:
            st.stop()

    st.markdown("### Solution")
    current_state = agent.graph.get_state(thread)
    # st.info(len(current_state.values['messages']))
    if current_state.next == ():
        last_message = current_state.values['messages'][-1]
        if len(last_message.invalid_tool_calls) > 0:
            last_message = current_state.values['messages'][-2]        
        st.info(f"{last_message.content}")
        

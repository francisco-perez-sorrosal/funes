import streamlit as st

import langchain_core.messages as lc_messages
from funes.agents.basic_agent import Role

def get_role_from_message(message):
    match (type(message)):
        case lc_messages.HumanMessage:
            st.write("Human Message")
            role = Role.USER
        case lc_messages.AIMessage:
            st.write("Tool Message")
            role = Role.ASSISTANT
        case lc_messages.ToolMessage:
            st.write("Tool Message")
            role = Role.ASSISTANT            
        case lc_messages.SystemMessage:
            st.write("System Message")
            role = Role.SYSTEM            
        case _:
            st.write(f"Unknown Message of type {type(message)}")
            role = Role.USER
    return role


def show_agent_messages(messages: list):
    for message in messages:
        role = get_role_from_message(message)
        st.chat_message(role).write(message)
    

def show_agent_state(state):
    with st.expander("Agent State"):
        st.title("Agent State")
        st.write(state.created_at)
        st.write("Next")    
        st.write(state.next)
        
        st.write("Configuration")
        st.write(state.config)
        st.write("Metadata")
        st.write(state.metadata)
        
        # st.write("Messages")
        # for msg in state:
        #     role = get_role_from_message(msg)
        #     st.chat_message(role).write(msg.content)
        # st.write("End of Messages")

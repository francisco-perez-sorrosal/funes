import streamlit as st

from typing import List, Tuple
from langchain_core.messages import HumanMessage, AIMessage

from rich import print

def format_chat_history(chat_history: List[Tuple[str, str]]):
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


def print_event(event: dict, _printed: set, max_length=1500, streamlit: bool = True):
    current_state = event.get("dialog_state")
    if current_state:
        if streamlit:
            st.info(f"Currently in: {current_state[-1]}")
        else:
            print(f"Currently in: {current_state[-1]}")
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=False)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            if streamlit:
                st.info(msg_repr)
            else:
                print(msg_repr)
            _printed.add(message.id)
import os
from typing import Optional
import streamlit as st

from funes.lang_utils import models, get_llm
from funes.agents.basic_agent import DoggieMultiAgent, Role, TOOLS, langchain_tools
from funes.prompt import basic_react_prompt, basic_template
from funes.agents.coordinator import ArxivRetrieverAgenticGraph
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from funes.ui.agent_state import show_agent_state, show_agent_messages
# from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
from langgraph.checkpoint.sqlite import SqliteSaver

from IPython.display import Image, display

from funes.ui.runner import AgentRunner
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



# # Helper functions
# # This substitutes tabs for now
# def segmented_control(labels: list[str], key: str, default: str | None = None, max_size: int = 6) -> str:
#     """Group of buttons with the given labels. Return the selected label."""
#     print(f"a = {key}")
#     if key not in st.session_state:
#         print("b")
#         st.session_state[key] = default or labels[0]
#     print("c")
#     selected_label = st.session_state[key]
#     print("d")
#     def set_label(label: str) -> None:
#         print("e")
#         st.session_state.update(**{key: label})
#     print("f")
#     cols = st.columns([1] * len(labels) + [max_size - len(labels)])
#     print("g")
#     for col, label in zip(cols, labels):
#         print("h")
#         btn_type = "primary" if selected_label == label else "secondary"
#         print(f"label= {label}")
#         col.button(label, on_click=set_label, args=(label,), use_container_width=True, type=btn_type)
#     print(f"selected label = {selected_label}")
#     return selected_label


st.title("Basic UI")

lm = st.session_state.get("lm")

if not lm:
    st.warning("No model selected")
    st.stop()



class AgentUI():
    
    def __init__(self, runner) -> None:
        self.runner = runner
        
    
    def get_snapshots(self,):
        new_label = f"thread_id: {self.runner.thread_id}, Summary of snapshots"
        sstate = ""
        for state in self.runner.agent.graph.get_state_history(self.runner.thread):
            for key in ['plan', 'draft', 'critique']:
                if key in state.values:
                    state.values[key] = state.values[key][:80] + "..."
            if 'content' in state.values:
                for i in range(len(state.values['content'])):
                    state.values['content'][i] = state.values['content'][i][:20] + '...'
            if 'writes' in state.metadata:
                state.metadata['writes'] = "not shown"
            sstate += str(state) + "\n\n"
        return new_label, sstate
    
    def update_snapshots(self):
        label, snapshots = self.get_snapshots()
        st.write(f"Nada: {label}")
        new_txt = st.text_area(label, value=snapshots)
        return new_txt
    
    
    def update_agent_node_ui(self, key):
        print(f"Get agent state for key: {key}")
        label, agent_state = self.runner.get_agent_state(key)
        print(f"Labelx: {label}, agent_state: {agent_state}")
        new_txt = st.text_area(label, value=agent_state)
        return new_txt

    


agent_runner = st.session_state.get("agent_runner")
if not agent_runner:
    agent_graph = DoggieMultiAgent(lm)
    agent_runner=AgentRunner(agent_graph)
    agent_ui = AgentUI(agent_runner)
    st.session_state["agent_runner"] = agent_runner
    st.session_state["agent_ui"] = agent_ui
    st.info(f"Agent {agent_runner.get_agent_name()} created and added to the session state")
else:
    st.info(f"Agent {agent_runner.get_agent_name()} found in the session state")


agent_ui = st.session_state.get("agent_ui")
if not agent_ui:
    st.error("An agent ui helper should be here but wasn't found! Fix this as it's a bug!")
    st.stop()

# tab_selected = segmented_control(["Agent", "Plan", "Snapshot"], default=None, key="control_tabs")

st.header("Agent")
st.image(agent_runner.agent.graph.get_graph().draw_png())

stop_after = st.multiselect("Options", ["planner", "planner_critic"])


def agent_invocation(user_prompt: Optional[str]):
    if user_prompt:
        st.chat_message(Role.USER).write(user_prompt)
        partial_message, lnode, nnode, thread_id, rev, acount = agent_runner.run_agent(True, user_prompt)
        st.chat_message(Role.ASSISTANT).write(f"Partial message: {partial_message}")


agent_tab, plan_tab, snapshot_tab = st.tabs(["Agent", "Plan", "Snapshot"])


messages = []
with agent_tab:
    user_prompt = st.text_area("Question", value="How much does a toy poodle weight?")

    if st.button('Submit', on_click=agent_invocation, args=(user_prompt,)):       
        st.info("First agent invocation")
        

    if st.button('Continue'):
        st.info("Subsequent agent invocation")
        partial_message, lnode, nnode, thread_id, rev, acount = agent_runner.run_agent(False, user_prompt)
        st.chat_message(Role.ASSISTANT).write(f"Partial message: {partial_message}")
        

with plan_tab:
    agent_ui.update_agent_node_ui("plan")

with snapshot_tab:
    st.header("State Snapshots")
    agent_ui.update_snapshots() 
   


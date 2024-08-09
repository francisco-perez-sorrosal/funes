import os
from collections.abc import Generator
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

from funes.ui.runner import AgentRunner, RunnerStateType, RunnerState
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
    st.session_state["agent_status"] = f"Agent {agent_runner.get_agent_name()} created and added to the session state"
else:
    st.session_state["agent_status"] = f"Agent {agent_runner.get_agent_name()} found in the session state"


agent_ui = st.session_state.get("agent_ui")
if not agent_ui:
    st.error("An agent ui helper should be here but wasn't found! Fix this as it's a bug!")
    st.stop()

# tab_selected = segmented_control(["Agent", "Plan", "Snapshot"], default=None, key="control_tabs")

st.sidebar.header("Agent")
st.sidebar.image(agent_runner.agent.graph.get_graph().draw_png())

### Status bars
agent_status = st.empty()
last_agent_status = st.session_state.get("agent_status", "No agent")

DEFAULT_STATUS = "Ready"
status_bar = st.empty()
last_status = st.session_state.get("last_status", DEFAULT_STATUS)

agent_col, status_col = st.columns(2)
with agent_col:
    agent_status.info(last_agent_status)

with status_col:
    status_bar.info(last_status)

stop_after = st.multiselect("Options", ["planner", "planner_critic"])

# st.info(f"Stop after: {stop_after}")

def invoke_agent_steps(user_prompt: Optional[str], start: bool):
    
    runner_state = agent_runner.run_agent(start, user_prompt, stop_after)
    invocation_no = 0
    while True:
        start = False
        if isinstance(runner_state, Generator):
            print("Generator")
            runner_state = next(runner_state, None)
        
        if runner_state:
            print(f"Runner state type: {runner_state.type.name}")
            match runner_state.type:
                case RunnerStateType.ITER:
                    runner_state = agent_runner.run_agent(start, user_prompt, stop_after)
                    status = f"Invocation {invocation_no}"
                    status_bar.info(status)
                    st.session_state["last_status"] = f"Invocation {invocation_no}"
                case RunnerStateType.END:
                    st.session_state["last_status"] = "Agent finished"
                    break
                case RunnerStateType.STOP:
                    st.session_state["last_status"] = "Agent stopped after step"
                    break
                case RunnerStateType.MAX_ITERS:
                    st.session_state["last_status"] = "Max iterations reached!"
                    break
        invocation_no += 1
        
agent_tab, plan_tab, snapshot_tab = st.tabs(["Agent", "Plan", "Snapshot"])


messages = []
with agent_tab:
    user_prompt = st.text_area("Question", value="How much does a toy poodle weight?")

    if st.button('Submit', on_click=invoke_agent_steps, args=(user_prompt, True)):
        st.info("First agent invocation")
        

    if st.button('Continue', on_click=invoke_agent_steps, args=(user_prompt, False)):
        st.info("Subsequent agent invocation")
        

with plan_tab:
    agent_ui.update_agent_node_ui("plan")

with snapshot_tab:
    st.header("State Snapshots")
    agent_ui.update_snapshots()

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

class AgentUI:
    
    def __init__(self, runner) -> None:
        self.runner: AgentRunner = runner
    
    def update_current_thread(self, ):
        thread_id = st.session_state.get("thread_id")
        st.write(f"Updating thread to {thread_id}")
        self.runner.set_current_thread(thread_id)
        st.session_state["last_status"] = f"New selected thread {thread_id}"
    
    def get_snapshots(self,):
        label_template = f"thread_id: {self.runner.current_thread_id()}, Snapshot: "
        state_list = []
        label_list = []
        current_thread_config = self.runner.current_thread_config()
        st.write(f"Current thread config: {current_thread_config}")
        for i, state in enumerate(self.runner.agent.graph.get_state_history(current_thread_config)):
            for key in ['plan']:
                if key in state.values:
                    state.values[key] = state.values[key][:80] + "..."
            if 'content' in state.values:
                for i in range(len(state.values['content'])):
                    state.values['content'][i] = state.values['content'][i][:20] + '...'
            if 'writes' in state.metadata:
                state.metadata['writes'] = "not shown"
            state_list.append(str(state) + "\n\n")
            label_list.append(f"{label_template} {i}")
        return label_list, state_list
    
    def update_snapshots(self):
        new_texts = {}
        labels, snapshots = self.get_snapshots()
        for label, snapshot in zip(labels, snapshots):
            new_texts[label] = st.text_area(label, value=snapshot, height=200)
        return new_texts
    
    def update_agent_node_ui(self, key):
        print(f"Get agent state for key: {key}")
        label, agent_state = self.runner.get_agent_state(key)
        print(f"Labelx: {label}, agent_state: {agent_state}")
        new_txt = st.text_area(label, value=agent_state, height=500)
        return new_txt
    
    def invoke_agent_steps(self, new_agent: bool, user_prompt: Optional[str], max_plan_revisions: int = 3, stop_after: list = []):
    
        if new_agent:
            st.info("Creating a new agent")
            self.runner.set_current_thread(None)
        runner_state = self.runner.run_agent(user_prompt, max_plan_revisions, stop_after)
        call_cnt = 0

        while True:
            
            if isinstance(runner_state, Generator):
                runner_state = next(runner_state, None)
            
            if runner_state:
                print(f"Runner state type: {runner_state.type.name}")
                match runner_state.type:
                    case RunnerStateType.ITER:
                        runner_state = self.runner.run_agent(user_prompt, max_plan_revisions, stop_after)
                        status = f"Invocation {call_cnt}"
                        status_bar.info(status)
                        st.session_state["last_status"] = f"Invocation {call_cnt}"
                    case RunnerStateType.END:
                        st.session_state["last_status"] = "Agent finished"
                        thread_id, final_resp = self.runner.get_final_response()
                        st.session_state["final_response"] = f"({thread_id}) Resp: {final_resp}"
                        break
                    case RunnerStateType.STOP:
                        st.session_state["last_status"] = "Agent stopped after step"
                        break
                    case RunnerStateType.MAX_ITERS:
                        st.session_state["last_status"] = "Max iterations reached!"
                        break
            call_cnt += 1


agent_runner = st.session_state.get("agent_runner")
if not agent_runner:
    agent_runner = AgentRunner(DoggieMultiAgent(lm))
    agent_ui = AgentUI(agent_runner)
    st.session_state["agent_runner"] = agent_runner
    st.session_state["agent_ui"] = agent_ui
    st.session_state["agent_status"] = f"Agent {agent_runner.get_agent_name()} created and added to the session state"
else:
    st.session_state["agent_status"] = f"Agent {agent_runner.get_agent_name()} found in the session state"

agent_started = agent_runner.is_started()

agent_ui = st.session_state.get("agent_ui")
if not agent_ui:
    st.error("An agent ui helper should be here but wasn't found! Fix this as it's a bug!")
    st.stop()

with st.sidebar:
    thread_id = st.selectbox("Select a thread", 
                             list(agent_runner.get_thread_ids()),
                             on_change=agent_ui.update_current_thread, key='thread_id',
                             index=0)
    select_thread_btn = st.button('Select Thread')
    st.write(thread_id)
    st.write(f"{type(thread_id)}")
    st.write(f"{select_thread_btn}")

st.sidebar.header("Agent")
st.sidebar.image(agent_runner.agent.graph.get_graph().draw_png())

### Status bars
DEFAULT_STATUS = "Ready"

agent_col, status_col = st.columns([1,1])
with agent_col:
    agent_status = st.empty()
    last_agent_status = st.session_state.get("agent_status", "No agent")
    last_agent_status = f"""{last_agent_status}\n
    - Started? {agent_started}\n
    - Thead: {agent_runner.current_thread_id()}"""
    agent_status.info(last_agent_status)

with status_col:
    status_bar = st.empty()
    last_status = st.session_state.get("last_status", DEFAULT_STATUS)    
    status_bar.info(last_status)
    if last_status == "Max iterations reached!":
        with st.popover("Open popover"):
            st.markdown("👋 Max iterations reached!")
            if st.button("Reset agent"):
                del st.session_state.lm
            st.button("Close")



        
agent_tab, plan_tab, snapshot_tab = st.tabs(["Agent", "Plan", "Snapshot"])

messages = []
with agent_tab:

    stop_after = st.multiselect("Stop after", ["planner", "planner_critic"])
    max_plan_revisions = st.number_input("Max plan revisions", value=3)
    user_prompt = st.text_area("Question", value="How much does a toy poodle weight?")
    final_resp = st.session_state.get("final_response", None)
    if final_resp is not None and final_resp != "":
        st.success(st.session_state["final_response"])

    col1, col2 = st.columns([1,1])
    with col1:
        if st.button('New agent', on_click=agent_ui.invoke_agent_steps, args=(True, user_prompt, max_plan_revisions, stop_after)):
            st.session_state["agent_status"] = "First agent invocation"
    with col2:
        if st.button('Continue', on_click=agent_ui.invoke_agent_steps, args=(False, user_prompt, max_plan_revisions, stop_after)):
            st.session_state["agent_status"] = "Subsequent agent invocation"        

with plan_tab:
    agent_ui.update_agent_node_ui("plan")

with snapshot_tab:
    st.header("State Snapshots")
    agent_ui.update_snapshots()

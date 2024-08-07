import os
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



# Helper functions
# This substitutes tabs for now
def segmented_control(labels: list[str], key: str, default: str | None = None, max_size: int = 6) -> str:
    """Group of buttons with the given labels. Return the selected label."""
    if key not in st.session_state:
        st.session_state[key] = default or labels[0]

    selected_label = st.session_state[key]

    def set_label(label: str) -> None:
        st.session_state.update(**{key: label})

    cols = st.columns([1] * len(labels) + [max_size - len(labels)])

    for col, label in zip(cols, labels):
        btn_type = "primary" if selected_label == label else "secondary"
        col.button(label, on_click=set_label, args=(label,), use_container_width=True, type=btn_type)

    return selected_label


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
        label, agent_state = self.runner.get_agent_state(key)
        new_txt = st.text_area(label, value=agent_state)
        return new_txt

    
class AgentRunner(object):

    def __init__(self, agent, max_iterations=10, share=False):
        self.agent = agent
        self.share = share
        self.partial_message = ""
        self.response = {}
        self.max_iterations = max_iterations
        self.iterations = []
        self.threads = []
        self.thread_id = -1
        self.thread = {"configurable": 
            {"thread_id": str(self.thread_id)}
        }
        st.write(f"Agent {self.get_agent_name()} created")


    def get_agent_name(self) -> str:
        return self.agent.name

    def get_display_state(self,):
        current_state = self.agent.graph.get_state(self.thread)
        lnode = current_state.values["lnode"]
        acount = current_state.values["count"]
        rev = current_state.values["revision_number"]
        nnode = current_state.next
        #print  (lnode,nnode,self.thread_id,rev,acount)
        return lnode, nnode, self.thread_id, rev,acount


    def get_agent_state(self, key):
        current_values = self.agent.graph.get_state(self.thread)
        if key in current_values.values:
            lnode, nnode, self.thread_id, rev, astep = self.get_display_state()
            new_label = f"last_node: {lnode}, thread_id: {self.thread_id}, rev: {rev}, step: {astep}"
            return new_label, current_values.values[key]
        else:
            return "", None


    def run_agent(self, start: bool, topic: str):
        #global partial_message, thread_id,thread
        #global response, max_iterations, iterations, threads
        if start:
            st.write("New agent")
            self.iterations.append(0)
            config = {'task': topic, 
                    'lnode': "", 
                    'planner': "no plan",
                    'revision_number': 0,
                    'max_revisions': 1,
                    'count':0}
            self.thread_id += 1  # new agent, new thread
            self.threads.append(self.thread_id)
        else:  
            st.write("Old agent")
            config = None  # This means continue execution when calling "invoke" below
        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}
        print(f"Invoking {self.get_agent_name()} for thread {self.thread_id}")
        
        while self.iterations[self.thread_id] < self.max_iterations:
            self.response = self.agent.graph.invoke(config, self.thread)
            self.iterations[self.thread_id] += 1
            self.partial_message += str(self.response)
            self.partial_message += f"\n------------------\n\n"
            lnode, nnode, _, rev, acount = self.get_display_state()
            return self.partial_message, lnode, nnode,self.thread_id, rev, acount
        return 


agent_runner = st.session_state.get("agent_runner")
if not agent_runner:
    st.write("Creating agent and runner")
    agent_graph = DoggieMultiAgent(lm)
    agent_runner=AgentRunner(agent_graph)
    agent_ui = AgentUI(agent_runner)
    st.session_state["agent_runner"] = agent_runner
    st.session_state["agent_ui"] = agent_ui
    st.info(f"Agent {agent_runner.get_agent_name()} created and added to the session state")
else:
    st.write("Agent and runner already created")
    st.info(f"Agent {agent_runner.get_agent_name()} found in the session state")


agent_ui = st.session_state.get("agent_ui")
if not agent_ui:
    st.error("An agent ui helper should be here but wasn't found! Fix this as it's a bug!")
    st.stop()

tab_selected = segmented_control(["Agent", "Plan", "Snapshot"], default="B", key="control_tabs")

messages = []
if tab_selected == "Agent":
    st.header("Agent")
    st.image(agent_runner.agent.graph.get_graph().draw_png())
    user_prompt = st.text_area("Question", value="How much does a toy poodle weight?")

    if st.button('Submit'):        
        st.chat_message(Role.USER).write(user_prompt)
        messages = [HumanMessage(content=user_prompt)]
        st.write("Invoking agent for the first time")
        agent_runner.run_agent(True, user_prompt)
    if st.button('Continue'):
        st.write("Subsequent agent invocation")
        agent_runner.run_agent(False, user_prompt)

elif tab_selected == "Plan":
    agent_ui.update_agent_node_ui("plan")

elif tab_selected == "Snapshot":
    st.header("State Snapshots")
    agent_ui.update_snapshots() 
   

# if user_prompt:
#     # # result=agent.graph.invoke({"messages": messages})  # Simple
#     # for event in agent.graph.stream({"messages": messages}, thread):
#     #     for evnt in event.values():
#     #         show_agent_messages(evnt)
#     st.write("Invoking agent for the first time")
#     agent.run_agent(True, user_prompt)
# else:
#     st.write("Subsequent agent invocation")
#     agent.run_agent(False, user_prompt)
#     # if agent:
#     #     st.write(f"### Last query events for thread {thread['configurable']['thread_id']}")
#     #     try:
#     #         for event in agent.graph.stream(None, thread):
#     #             for evnt in event.values():
#     #                 # st.write(evnt)
#     #                 # st.write(type(evnt))
#     #                 show_agent_messages(evnt["messages"])
#     #     except ValueError as e:
#     #         st.warning(e)
            


# if agent:
#     while agent.graph.get_state(thread).next:
        
#         agent_state = agent.graph.get_state(thread)
#         # st.write(type(agent_state))
#         show_agent_state(agent_state)
#         # st.chat_message(role).write(agent_state.values)
#         proceed_btn = st.button(f"proceed? ({agent_state.created_at})", key=agent_state.created_at)
#         if proceed_btn:
#             st.write("Proceed pressed")
#             for event in agent.graph.stream(None, thread):
#                 for v in event.values():
#                     st.write(v)
#         else:
#             st.stop()

#     st.markdown("### Solution")
#     current_state = agent.graph.get_state(thread)
#     # st.info(len(current_state.values['messages']))
#     if current_state.next == ():
#         last_message = current_state.values['messages'][-1]
#         if len(last_message.invalid_tool_calls) > 0:
#             last_message = current_state.values['messages'][-2]        
#         st.info(f"{last_message.content}")
        

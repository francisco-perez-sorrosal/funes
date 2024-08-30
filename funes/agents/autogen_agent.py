import os

import chainlit as cl

from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, ConversableAgent

with open("openai", "r") as file:
    openai_api_key = file.read().strip()
# st.write(f"Open API key: {openai_api_key}")
os.environ["OPEN_API_KEY"] = openai_api_key

def chat_new_message(self, message, sender):
    cl.run_sync(
        cl.Message(
            content="",
            author=sender.name,
        ).send()
    )
    content = message.get("content")
    cl.run_sync(
        cl.Message(
            content=content,
            author=sender.name,
        ).send()
    )
    

def config_personas():
    config_list = [{
        "model": "gpt-4o-mini",  # model name
        "api_key": openai_api_key  # api key
    }]
    llm_config = {
        "seed": 14,  # seed for caching and reproducibility
        "config_list": config_list,  # a list of OpenAI API configurations
        "temperature": 0.7,  # temperature for sampling
    }
    
    user_proxy = UserProxyAgent(
            name="User_Proxy",
            system_message="A human director.",
            max_consecutive_auto_reply=2,
            llm_config=llm_config,
            human_input_mode="NEVER"
        )
    
    content_creator = AssistantAgent(
        name="Content_Creator",
        system_message="I am a content creator that talks about exciting technologies about AI. "
                       "I want to create exciting content for my audience that is about the latest AI technology. "
                       "I want to provide in-depth details of the latest AI white papers.",
        llm_config=llm_config,
    )
    
    script_writer = AssistantAgent(
        name="Script_Writer",
        system_message="I am a script writer for the Content Creator. "
                       "This should be an eloquently written script so the Content Creator can "
                       "talk to the audience about AI.",
        llm_config=llm_config
    )
    
    researcher = AssistantAgent(
        name="Researcher",
        system_message="I am the researcher for the Content Creator and look up the latest white papers in AI."
                       " Make sure to include the white paper Title and Year it was introduced to the Script_Writer.",
        llm_config=llm_config
    )
    
    reviewer = AssistantAgent(
        name="Reviewer",
        system_message="I am the reviewer for the Content Creator, Script Writer, and Researcher once they are done "
                       "and have come up with a script.  I will double check the script and provide feedback.",
        llm_config=llm_config
    )

    group_chat = GroupChat(
        agents=[user_proxy, content_creator, script_writer, researcher, reviewer], messages=[]
    )


    manager = GroupChatManager(groupchat=group_chat, llm_config=llm_config)
    return user_proxy, manager


def start_chat_script(message, output_to_console=False):
    if output_to_console:
        ConversableAgent._print_received_message = chat_new_message
    user_proxy, manager = config_personas()
    user_proxy.initiate_chat(manager, message=message)
    
    
if __name__ == "__main__":
    test_message = ("I need to create a YouTube Script that talks about the latest paper about gpt-4 on arxiv and its "
                    "potential applications in software.")
    start_chat_script(test_message, output_to_console=True)

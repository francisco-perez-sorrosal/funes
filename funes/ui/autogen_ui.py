import chainlit as cl

from funes.agents.autogen_agent import start_chat_script

@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="Funes the Memorious",
            markdown_description="Your next YouTube video script is just a few messages away!",
        ),
        cl.ChatProfile(
            name="SaaS Product Ideation",
            markdown_description="Get your next SaaS product idea in a few messages!",
        ),
    ]
@cl.on_chat_start
async def on_chat_start():
    chat_profile = cl.user_session.get("chat_profile")
    await cl.Message(
        content=f"Welcome to {chat_profile} chat. Please type your first message to get started."
    ).send()


@cl.on_message
async def on_message(message):
    print(message)
    chat_profile = cl.user_session.get("chat_profile")
    print(chat_profile)
    message_content = message.content
    if chat_profile == "Funes the Memorious":
        start_chat_script(message_content)
    else:
        await cl.Message(
            content="I am sorry, That profile doesn't exist your request."
        ).send()

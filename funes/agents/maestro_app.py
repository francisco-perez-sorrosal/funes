def reflection_message(recipient, messages, sender, config):
    print(f"{recipient} reflecting... (Caller: {sender})", "yellow")
    return f'''Review the following content. 
            \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}'''






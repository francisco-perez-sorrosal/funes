import os
from enum import Enum
import re
from typing import Dict
from langchain_huggingface import ChatHuggingFace
from pydantic import BaseModel


def calculate(what):
    return eval(what)

def average_dog_weight(name):
    if name in "Scottish Terrier": 
        return("Scottish Terriers average 20 lbs")
    elif name in "Border Collie":
        return("a Border Collies average weight is 37 lbs")
    elif name in "Toy Poodle":
        return("a toy poodles average weight is 7 lbs")
    else:
        return("An average dog weights 50 lbs")

TOOLS = {
    "calculate": calculate,
    "average_dog_weight": average_dog_weight
}


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

def msg_builder(content:str, role:Role = Role.USER):
    return {"role": role, "content": content}

action_re = re.compile('^Action: (\w+): (.*)$') 


class BasicAgent:
    def __init__(self, llm, system_msg:str = ""):
        self.llm = llm
        self.chat = ChatHuggingFace(llm=self.llm, model_id="meta-llama/Meta-Llama-3-8B-Instruct")
        self.system_msg = system_msg
        self.messages = []
        if self.system_msg:
            self.messages.append(msg_builder(self.system_msg, Role.SYSTEM))
                                 
    def __call__(self, msg):
        self.messages.append(msg_builder(msg))
        result = self.execute()
        self.messages.append(msg_builder(result, Role.ASSISTANT))
        return result
    
    def execute(self):
        completion = self.chat.invoke(self.messages)
        print(completion.content)
        return completion.content

        # completion = self.llm.chat.completions.create(
        #     model="gpt-4o",
        #     temperature=0,
        #     messages=self.messages,
        # )
        # return completion.choices[0].message.content


    def query(self, question, known_tools: Dict[str, str], max_turns=5):
        i = 0
        next_prompt = question
        result = ""
        while i < max_turns:
            i += 1
            result = self(next_prompt)
            print("JFSDJFD")
            print(result)
            actions = [
                action_re.match(a) 
                for a in result.split('\n') 
                if action_re.match(a)
            ]
            if actions:
                # There is an action to run
                action, action_input = actions[0].groups()
                if action not in known_tools:
                    raise Exception("Unknown action: {}: {}".format(action, action_input))
                print(" -- running {} {}".format(action, action_input))
                observation = known_tools[action](action_input)
                print("Observation:", observation)
                next_prompt = "Observation: {}".format(observation)
            else:
                if result and "Answer" in result:
                    return result
                else:
                    return "No result found"

            


    
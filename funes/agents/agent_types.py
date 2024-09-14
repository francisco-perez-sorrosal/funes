import json

from typing import Any, Callable, Dict, List, Literal, Optional, Union
from enum import Enum


from autogen.agentchat import ConversableAgent, GroupChat, AssistantAgent, UserProxyAgent, GroupChatManager
from pydantic import BaseModel
from pathlib import Path


from llm_foundation import logger


DEFAULT_TERMINATION_WORD="TERMINATE"

def is_termination_message(msg, termination_word: str = DEFAULT_TERMINATION_WORD) -> bool:
    '''Detects if we should terminate the conversation'''
    
    if isinstance(msg.get("content"), str):
        return msg["content"].rstrip().endswith(termination_word)
    elif isinstance(msg.get("content"), list):
        for content in msg["content"]:
            if isinstance(content, dict) and "text" in content:
                return content["text"].rstrip().endswith(termination_word)
    return False

def always_terminate(msg) -> bool:
    return True


def custom_speaker_selection_func(last_speaker, groupchat):
    """Define a customized speaker selection function.
    A recommended way is to define a transition for each speaker in the groupchat.

    Parameters:
        - last_speaker: Agent
            The last speaker in the group chat.
        - groupchat: GroupChat
            The GroupChat object
    Return:
        Return one of the following:
        1. an `Agent` class, it must be one of the agents in the group chat.
        2. a string from ['auto', 'manual', 'random', 'round_robin'] to select a default method to use.
        3. None, which indicates the chat should be terminated.
    """
    pass


class AutogenAgentType(Enum):
    ConversableAgent = 1
    UserProxyAgent = 2
    AssistantAgent = 3
    GroupChatManager = 4
    

class Role(BaseModel):
    description: str
    agent_system_message: str
    human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"] = "TERMINATE"
    autogen_code_execution_config: dict = {}

    @classmethod
    def from_json_file(cls, file_path: str) -> 'Role':
        with open(file_path, 'r') as file:
            json_data = json.load(file)
        return cls(**json_data)
    
    def to_autogen_agent(self, 
                         name:str, 
                         type: AutogenAgentType, 
                         human_input_mode: Optional[Literal["ALWAYS", "NEVER", "TERMINATE"]] = None,
                         llm_config: Optional[dict] = None,
                         max_consecutive_auto_reply: Optional[int] = None,
                         group_chat: Optional[GroupChat] = None,
                         code_execution_config: Optional[Union[Dict, Literal[False]]] = False,
                         termination_function: Optional[Callable] = None) -> 'ConversableAgent':
        
        code_execution_config = code_execution_config if code_execution_config is not None else self.autogen_code_execution_config
        
        if termination_function is not None:
            termination_message_placeholder = lambda msg: termination_function(msg)
        else:
            termination_message_placeholder = None
        
        match type:
            case AutogenAgentType.ConversableAgent:
                return ConversableAgent(name=name, 
                                                system_message=self.agent_system_message,
                                                description=self.description,
                                                llm_config=llm_config,
                                                human_input_mode=human_input_mode if human_input_mode is not None else self.human_input_mode,
                                                max_consecutive_auto_reply=max_consecutive_auto_reply,
                                                is_termination_msg=termination_message_placeholder,
                                                code_execution_config=code_execution_config
                )
            case AutogenAgentType.AssistantAgent:
                return AssistantAgent(name=name, 
                                              system_message=self.agent_system_message, 
                                              description=self.description,
                                              llm_config=llm_config,
                                              human_input_mode=human_input_mode if human_input_mode is not None else self.human_input_mode,
                                              max_consecutive_auto_reply=max_consecutive_auto_reply,
                                              is_termination_msg=termination_message_placeholder,
                                              code_execution_config=code_execution_config,
                )
            case AutogenAgentType.UserProxyAgent:
                return UserProxyAgent(name=name, 
                                              system_message=self.agent_system_message, 
                                              description=self.description,
                                              llm_config=llm_config,
                                              human_input_mode=human_input_mode if human_input_mode is not None else self.human_input_mode,
                                              max_consecutive_auto_reply=max_consecutive_auto_reply,
                                              is_termination_msg=termination_message_placeholder,
                                              code_execution_config=code_execution_config,
                )
            case AutogenAgentType.GroupChatManager:
                if group_chat is None:
                    raise ValueError("Group chat is required for GroupChatManager")
                return GroupChatManager(name=name, 
                                                groupchat=group_chat,
                )
            case _:
                raise ValueError(f"Invalid agent type: {type}")


class Persona(BaseModel):
    name: str
    roles: Dict[str, Role]
    
    @classmethod
    def from_json_file(cls, file_path: str) -> 'Persona':
        with open(file_path, 'r') as file:
            json_data = json.load(file)
        return cls(**json_data)

    def save_as_json(self, 
                     path: Optional[Path] = None, 
                     file_name: Optional[str] = None, 
                     overwrite_existing: bool = True):
        if path is None:
            path = Path(self.__class__.__name__)
        if file_name is None:
            file_name = self.name
        if not file_name.endswith(".json"):
            file_name += ".json"
        path = path / file_name
        
        # Create the directory if it does not exist
        path.parent.mkdir(parents=True, exist_ok=overwrite_existing)
        
        with open(path, 'w') as file:
            file.write(self.model_dump_json(indent=4))
            
            
        
        logger.info(f"JSON object written: {path}")
        
    def get_roles(self) -> List[str]:
        return list(self.roles.keys())

    def role_to_autogen_agent(self,
                              role_name: str,
                              type: AutogenAgentType, 
                              human_input_mode:Literal["ALWAYS", "NEVER", "TERMINATE"] = "NEVER",
                              llm_config: Optional[dict] = None,
                              group_chat: Optional[GroupChat] = None,
                              code_execution_config: Optional[Union[Dict, Literal[False]]] = False,
                              termination_function: Optional[Callable] = None) -> 'ConversableAgent':
        
        return self.roles[role_name].to_autogen_agent(name=f"{role_name}", 
                                                       type=type, 
                                                       human_input_mode=human_input_mode,
                                                       llm_config=llm_config,
                                                       group_chat=group_chat,
                                                       code_execution_config=code_execution_config,
                                                       termination_function=termination_function)

    def __str__(self) -> str:
        def format_role(role_name: str, role: Role, indent: int) -> str:
            indent_str = " " * indent
            role_str = f"{indent_str}Role: {role_name}\n"
            role_str += f"{indent_str}Description: {role.description}\n"
            role_str += f"{indent_str}Agent System Message: {role.agent_system_message}\n"
            role_str += f"{indent_str}Autogen Code Execution Config: {role.autogen_code_execution_config}\n"
            return role_str

        def format_persona(persona: Persona, indent: int) -> str:
            indent_str = " " * indent
            persona_str = f"{indent_str}Persona: {persona.name}\n"
            for role_name, role in persona.roles.items():
                persona_str += format_role(role_name, role, indent + 2)
            return persona_str

        return format_persona(self, 0)
    
    
class Application(BaseModel):
    name: str
    roles: Dict[str, Role]
    
    @classmethod
    def from_json_file(cls, file_path: str) -> 'Application':
        with open(file_path, 'r') as file:
            json_data = json.load(file)
        return cls(**json_data)
    
    
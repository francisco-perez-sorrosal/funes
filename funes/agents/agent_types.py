import json

from typing import Dict, List, Literal, Optional
from enum import Enum

import autogen
from autogen.agentchat import Agent, GroupChat
from pydantic import BaseModel
from pathlib import Path


from llm_foundation import logger


class AutogenAgentType(Enum):
    ConversableAgent = 1
    UserProxyAgent = 2
    AssistantAgent = 3
    GroupChatManager = 4
    

class Role(BaseModel):
    description: str
    agent_system_message: str
    autogen_code_execution_config: dict = {}

    @classmethod
    def from_json_file(cls, file_path: str) -> 'Role':
        with open(file_path, 'r') as file:
            json_data = json.load(file)
        return cls(**json_data)
    
    def to_autogen_agent(self, 
                         name:str, 
                         type: AutogenAgentType, 
                         human_input_mode:Literal["ALWAYS", "NEVER", "TERMINATE"] = "NEVER",
                         llm_config: Optional[dict] = None,
                         group_chat: Optional[GroupChat] = None,
                         code_execution_config: Optional[dict] = None) -> 'Agent':
        
        code_execution_config = code_execution_config if code_execution_config is not None else self.autogen_code_execution_config
        
        match type:
            case AutogenAgentType.AssistantAgent:
                return autogen.AssistantAgent(name=name, 
                                              system_message=self.agent_system_message, 
                                              llm_config=llm_config,
                                              human_input_mode=human_input_mode,
                                              code_execution_config=code_execution_config
                                              )
            case AutogenAgentType.ConversableAgent:
                return autogen.ConversableAgent(name=name, 
                                                system_message=self.agent_system_message,
                                                code_execution_config=code_execution_config,
                                                llm_config=llm_config)
            case AutogenAgentType.UserProxyAgent:
                return autogen.UserProxyAgent(name=name, 
                                              system_message=self.agent_system_message, 
                                              llm_config=llm_config,
                                              code_execution_config=code_execution_config,
                                              human_input_mode="ALWAYS",)
            case AutogenAgentType.GroupChatManager:
                if group_chat is None:
                    raise ValueError("Group chat is required for GroupChatManager")
                return autogen.GroupChatManager(name=name, 
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
                              code_execution_config: Optional[dict] = None) -> 'Agent':
        
        return self.roles[role_name].to_autogen_agent(name=f"{self.name}_{role_name}", 
                                                       type=type, 
                                                       human_input_mode=human_input_mode,
                                                       llm_config=llm_config,
                                                       group_chat=group_chat,
                                                       code_execution_config=code_execution_config)

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
    
    
# Class Late Binding Annotaion
from __future__ import annotations 

# Import Python Native Lib
import os, sys
import keyboard
import asyncio
import numpy as np
import uuid as uuid
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from typing_extensions import Literal, TypedDict

# Import LangGraph & LangChain packages
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, AnyMessage, RemoveMessage

# Import Langchain AI Models
from langchain_ollama import ChatOllama
#from langchain_openai import ChatOpenAI

# Load Env Variables
load_dotenv()


# Intent Message Dict
class IntentState(TypedDict):
    int_message: list[AnyMessage]

class UserIntentPrompt(BaseModel):
    user_prompt: str = Field(description="User Prompt", default_factory=str)


# Class used to Check Intent of User's Request
class UserIntent:

    def __init__(self, system_instr: str, check_intent_docstr: str):
        self.deployable_list = None
        self.intentchecker_llm = None
        self.system_instruction = system_instr
        self.check_intent_docstring = check_intent_docstr

    # Create Intent Checking - GPT4.1
    def instantiate_llm(self, model_name: str) -> object:

        #self.intentchecker_llm = ChatOpenAI(
        #    model=model_name, 
        #    temperature=0.0, 
        #    max_retries=2
        #)

        self.intentchecker_llm = ChatOllama(
            model = model_name,
            temperature = 0.0,
            num_predict = 8192,
            num_ctx = 8192,
            model_kwargs = {
                "repeat_penalty": 1.1,
                "options": {
                    "use_cache": False,
                    "use_mmap": False
                }
            }   
        )

    # Set Deployable Artifact List
    def set_deployable_artifacts(self, artifact_list: str):
        try:
            str_array = artifact_list.split(',')

            self.deployable_list = []
            for item in str_array:
                self.deployable_list.append(item.strip())

        except Exception as exp:
            error = f"Error inside UserIntent.set_deployable_artifacts: {exp}"
            print(error, flush=True)
            raise Exception(error)

    # Specs for Tool in UserIntent Class
    async def get_tool_spec(self, tool_name: str) -> dict:
        try:
            # Check Intent Spec
            if (tool_name == "check_intent"):
                tool_spec = {
                    "name": "check_intent",
                    "description": self.check_intent_docstring,
                    "args_schema": UserIntentPrompt
                }

            return(tool_spec)
        
        except Exception as exp:
            error = f"Error inside UserIntent.get_tool_spec: {exp}"
            print(error, flush=True)
            raise Exception(error)

    # Check for User Intent Tool
    async def check_intent(self, user_prompt: str) -> str:
        try:

            print("Inside Intent Tool Call", flush=True)

            intent_llm = self.intentchecker_llm
            system_prompt_intent = self.system_instruction + "\n\nIntent List:" + str(self.deployable_list)

            intent_array = []
            intent_array = [SystemMessage(content=system_prompt_intent),
                            HumanMessage(content=user_prompt)
            ]

            intent_prompt: IntentState = IntentState({"int_message": intent_array})
            ai_response = intent_llm.invoke(intent_prompt["int_message"])
            response = ai_response.content

            print("Intent: ", response, flush=True)

            return(response)
        
        except Exception as exp:
            error = f"Error inside UserIntent.check_intent: {exp}"
            print(error, flush=True)
            raise Exception(error)
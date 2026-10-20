
# This Class provides is an LLM (ChatOllama) based Template 
#   Substitution capability.
# This is used to Substitute User provided Values into
#   respective Cloud Formation Template thereby creating an
#   instance of the CFT for deployment.
# MCP Tool 'update_template' calls this Class to do the heavylifting
#   of Template Substitution.


# Class Late Binding Annotaion
from __future__ import annotations 

# Import Python Native Lib
import os, sys
import json
import asyncio
import numpy as np
import uuid as uuid
from dotenv import load_dotenv
from string import Template

from pydantic import BaseModel, Field
from typing_extensions import Literal, TypedDict

# Import Langchain Core & AI Models
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AnyMessage

# Load Env Variables
load_dotenv()

# AWS Infra Template Substitute Model 
model = os.getenv("mcp_template_sub_model")


# Class Used to Substitute User Provided Data
# For AWS Cloudformation Launch Template
class TemplateDataSubstitution:

    def __init__(self, template_sub_llm_instr: str, sub_content_format_instr: str, baked_template_ex: str):
        self.template_sub_llm_instr = template_sub_llm_instr
        self.template_sub_content_instr = sub_content_format_instr
        self.baked_template_example = baked_template_ex

        self.substitution_llm = ChatOllama(
            model = model,
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

    # Check for User Intent Tool
    def do_substitution(self, template_input: dict, template_file_content: str) -> str:
        try:

            print("\nInside CloudFormation Template Substitution", flush=True)

            substitution_llm = self.substitution_llm
            system_prompt = self.template_sub_llm_instr + self.baked_template_example

            # Get & Format User and File Content
            user_input = json.dumps(template_input)
            file_content = template_file_content
            str_template = Template(self.template_sub_content_instr)
            content = str_template.substitute(user_input=user_input, file_content=file_content)

            content_array = []
            content_array = [SystemMessage(content=system_prompt),
                            HumanMessage(content=content)
            ]

            # Invoke LLM for Substitution
            ai_response = substitution_llm.invoke(content_array)
            response = ai_response.content

            return(response)
        
        except Exception as exp:
            error = f"Error inside TemplateDataSubstitution.do_substitution: {exp}"
            print(error, flush=True)
            raise Exception(error)
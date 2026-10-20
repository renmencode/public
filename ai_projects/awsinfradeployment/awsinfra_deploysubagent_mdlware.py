# Sub-Agent Middleware.

# This Middleware is used to manipulate SubAgent Tool Calls Events.
# Though File Operation are inherent to SubAgent, that is 
#   prevented with instructions to skip unintended file calls. 
# This Middleware also implements HTIL where on 'deploy_template'
#   HTIL is triggered to accept StackName for the dpeloyemnt.
# Rest of the Tool Calls that goes thruthe Middleware are 
#    - artifact_read_template, Update_template, deploy_tempate. 


# Class Late Binding Annotaion
from __future__ import annotations 

# Import Python Native Lib
import os, sys
import json
import asyncio
import uuid as uuid
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from collections.abc import Awaitable, Callable
from typing_extensions import Annotated, TypedDict, Literal

# Import Langchain Modules
from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command, interrupt
from langgraph.errors import GraphInterrupt

# Load Env Vars
load_dotenv()


# Primary Agent Middleware Class
class DeploySubAgentMiddleware(AgentMiddleware):

    def __init__(self, middleware_instr: dict):
        self.middleware_instr = middleware_instr

    # Tool Call Middleware Asyc Function
    async def awrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]) -> ToolMessage | Command:
        try:
            tool_resp = None
            tool_call_id = request.tool_call["id"]
            tool_call_name = request.tool_call["name"]
            tool_call_args = request.tool_call["args"]

            # Skipping SubAgent Native File Operations
            # This cannot be removed since it is Defaut Middleware for Agents
            # Only way to skip is short circuit the call.
            system_file_ops_name = ["ls", "glob", "read_file", "write_file", "edit_file", "grep", "execute"]
            if (tool_call_name not in system_file_ops_name):
                #print("\nSUB-AGENT MW-TOOL CALL REQUEST HANDLER:", request)

                if (tool_call_name == "deploy_template"):
                    deployment_stack_name = (tool_call_args["user_input"]).pop("stack_name")    # User Interrupt Input 'stack_name' Extract & Remove.
                    baked_launch_template = request.state.get("baked_launch_template")          # Get Baked Template from State
                    request.tool_call["args"]["deployment_stack_name"] = deployment_stack_name              
                    request.tool_call["args"]["yaml_template_instance"] = baked_launch_template
                else:
                    pass

                response = await handler(request)       # Call Respective Tool

            else:
                #print("\nSUB-AGENT FILESYSTEM MW-TOOL HANDLER-ELSE:", request)
                tool_content = self.middleware_instr["subagent_fileops_skip_instruction"]
                response = ToolMessage(content=tool_content, name=tool_call_name, tool_call_id=tool_call_id)

            #print("\nSub-Agent ToolCall ID:", tool_call_id, flush=True)
            #print("\nSub-Agent ToolCall NAME:", tool_call_name, flush=True)
            #print("\nSub-Agent Response TYPE:", type(response), flush=True)
            #print("\nSub-Agent Original Response:", response, flush=True)

            # Process Response from Tool Calls.
            if isinstance(response, Command):
                tool_resp = response
                #print("\nInside SubAgent Middlweare Command:\n", flush=True)
            elif isinstance(response, ToolMessage):
                # Check Tool Execution Error
                # And Re-Throw Exception
                if (response.status == "error"):
                    raise Exception((response.content[0])["text"])

                # Put Baked Template into State using Command.
                if (tool_call_name == "update_template"):
                    update_template_output = (response.content[0])["text"]
                    tool_message = ToolMessage(content="Launch Template Update Successful", name=tool_call_name, tool_call_id=tool_call_id)
                    tool_resp = Command(
                        update={
                            "baked_launch_template": update_template_output,
                            "messages": [tool_message]
                        }
                    )
                elif (tool_call_name == "deploy_template"):
                    deployment_output = (response.content[0])["text"]
                    resp_content = f"Artifact Deployment Successful. Deployment Output Attributes: {deployment_output}"
                    tool_resp = ToolMessage(content=resp_content, name=tool_call_name, tool_call_id=tool_call_id)
                else:
                    tool_resp = response
                    
                #print("\nInside SubAgent Middlweare ToolMessage:\n", flush=True)
            else:
                raise Exception(f"Unknown ToolCall Response Type (Nither Command or ToolMessage). Executed Tool is - {tool_call_name}.")

            return(tool_resp)
        
        except GraphInterrupt:          # Exception Type used to boil up Interrupt to Parent Agent
            raise
        except Exception as exp:
            error = f"Error inside DeploySubAgentMiddleware.awrap_tool_call: {exp}"
            print(error, flush=True)
            raise Exception(error)

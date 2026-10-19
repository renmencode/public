# Primary Agent Middleware.

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
class DeployAgentMiddleware(AgentMiddleware):

    def __init__(self, middleware_instr: dict):
        self.middleware_instr = middleware_instr

    # Tool Call Middleware Asyc Function
    async def awrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]) -> ToolMessage | Command:
        try:
            tool_resp = None
            tool_call_id = request.tool_call["id"]
            tool_call_name = request.tool_call["name"]

            # Skipping Agent Native File Operations
            # This cannot be removed since it is Defaut Middleware for Agents
            # Only way to skip is short circuit the call.
            system_file_ops_name = ["ls", "glob", "read_file", "write_file", "edit_file", "grep", "execute"]
            if (tool_call_name not in system_file_ops_name):
                #print("\nMW-TOOL CALL REQUEST:", request)

                response = await handler(request)

            else:
                #print("\nAGENT FILESYSTEM MW-TOOL HANDLER-ELSE:", request)
                tool_content = self.middleware_instr["agent_fileops_skip_instruction"]
                response = ToolMessage(content=tool_content, name=tool_call_name, tool_call_id=tool_call_id)


            #print("\n ToolCall NAME:", tool_call_name, flush=True)
            #print("\n ToolCall TYPE:", type(response), flush=True)
            #print("\n ToolCall VALUE:", response, flush=True)

            # Process Response from Tool Calls.
            if isinstance(response, Command):
                if (tool_call_name == "task"):
                    tool_resp = response
                #print("\nInside PArent Agent Middlweare Command:\n", flush=True)
            elif isinstance(response, ToolMessage):
                # Check Tool Execution Error
                # And Re-Throw Exception
                if (response.status == "error"):
                    raise Exception((response.content[0])["text"])

                # Process Response from Tool Calls.
                if (tool_call_name == "check_intent"):
                    #print("\nInside CheckIntent ToolCall:\n", response, flush=True)
                    check_intent_instr = self.middleware_instr["check_intent"]
                    tool_content = response.content + "." + check_intent_instr
                    tool_resp = ToolMessage(content=tool_content, tool_call_id=tool_call_id)
                else:
                    tool_resp = response                # Default if Tools other than Check Intent (for future)

                #print("\nInside Parent Agent Middlweare ToolMessage:\n", flush=True)
            else:
                raise Exception(f"Unknown ToolCall Mismatch. Executed Tool is - {tool_call_name}.")

            return(tool_resp)

        except GraphInterrupt:          # Exception Type used to boil up Interrupt to LangGraph Engine
            raise
        except Exception as exp:
            error = f"Error inside DeployAgentMiddleware.awrap_tool_call: {exp}"
            print('\n' + error, flush=True)
            return ToolMessage(content=error, name=tool_call_name, tool_call_id=tool_call_id)

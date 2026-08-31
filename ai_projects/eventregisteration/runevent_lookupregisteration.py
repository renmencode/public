# Notes:
# Registration Lookup Sub-Agent. This is invoked from Run Event Agent.
#
# Functionality:
# 1. Agent Technology - LangChain
# 2. LLM Used - Qwen1.3-8B 4K_M Quantized
# 3. Input : Exact Prompt that User input. 
# 4. Output: List or Individual Registration Information.
# 5. Tools: List User, Lookup User publised as MCP Services.
# 6. Point of Entry - 'initialize' and then 'lookup'.
# 7. System Instruction: runningevent_agent_instructions.json 
#           #registeration_lookup - Primary System Instructions for Sub-Agent
#           #registeration_lookup_response - Instructions attached as Response of 'lookup' ToolCall
#                                            to parent Run Event Agent.
#           #lookup_toolcall_response - Instructions attached as Response to ToolCall (list_users 
#                                       or lookup_user) made by Sub-Agent.
#
# Proces Flow:
# 1. This Sub-Agent is created using LangChain's 'create_agent' method.
# 2. We attach Model (Ollama), Tools, Middleware, and System Instructions when creating Sub-Agent.
# 3. Middleware is created as wrapper for the tools - wrap_tool_call
#    - This is used to manipulate the ToolCall and its Response for formatting purpose
#    - Functionally it is very similar to having a Custom ToolNode Executor as implemented for LangGraph.
# 4. If the Request is to List all Registred Users - ths Sub-Agent calls the list_user MCP Endpoint.
# 5. If the Request is to Lookup a specific Registred User using RegisterationId - the Sub-Agent calls
#    the lookup_user MCP Endpoint.
# 6. Response from the ToolCall is send back to LLM with Instructions injected inside 'wrap_tool_call' methods.
# 7. Since the Middleware is associated with Sub-Agent, all ToolCalls and its Responses to LLM are routed in
#    LangChain via this middleware. 
#    - It is the Responsibility of teh wrapper to invoke the actual Tool based on the 'Request Object'
#    provied by LLM. 
#    - Ona side note we can chain multiple Middleware functions as we do with Web Servlets.


# Class Late Binding Annotaion
from __future__ import annotations 

# Import Python Native Lib
import os, sys
import json
import numpy as np
import uuid as uuid
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from collections.abc import Awaitable, Callable
from typing_extensions import Annotated, TypedDict, Literal

# Import Langchain packages
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import StructuredTool
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, AnyMessage
from langgraph.types import Command
from langsmith import traceable

# Import Langchain ChatOllama
from langchain_ollama import ChatOllama

# Import Custom Class
from runevent_mcpclient import MCPClient

# Load Env Variables
load_dotenv()

# Init Langsmith Tracing
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("runevent_monitor_app")


# Input Schema for Registeration Lookup
class RegisterationLookupInput(BaseModel):
    user_prompt: str = Field(description="User Prompt", default_factory=str)


# RegisterationLookup LLM
class RegisterationLookupAgentLLM:

    def __init__(self):
        pass

    def create_reglookup_llm(self, model_name: str, tools: list[StructuredTool]):
        try:
            ol_reglkp_llm = ChatOllama(
                model = model_name,
                temperature = 0.0,
                num_predict = 8192,
                num_ctx = 2048,
                model_kwargs = {
                    "repeat_penalty": 1.1,
                    "options": {
                        "use_cache": False,
                        "use_mmap": False
                    }
                }   
            )
            ol_reglkp_llm_with_tools = ol_reglkp_llm.bind_tools(tools)

            return(ol_reglkp_llm_with_tools)

        except Exception as exp:
            error = f"Error inside RegisterationLookupAgentLLM.create_reglookup_llm - {exp}"
            print(error, flush=True)
            raise Exception(error)


# RegisterationLookup Agent Middleware
class RegisterationLookupAgentMiddleware(AgentMiddleware):

    def __init__(self, tool_instr: str):
        self.tool_resp_instr = tool_instr

    # Agent does ToolCall thru the Middleware Wrapper
    async def awrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]):
        try:
            tool_call_id = request.tool_call["id"]
            tool_call_name = request.tool_call["name"]

            tool_resp = await handler(request)

            # Check if Resonse is an Error Message
            # If No Error Format Tool Message with Instr to LLM for Response Format.
            content = json.loads((tool_resp.content[0])["text"])
            if (content["error_message"] is not None):
                response = ToolMessage(content=content["error_message"], tool_call_id=tool_call_id)

            elif (tool_call_name == "list_users"):
                del content["error_message"]
                response = ToolMessage(content=json.dumps(content) + "." + self.tool_resp_instr, 
                                       tool_call_id=tool_call_id
                                    )
            elif (tool_call_name == "lookup_user"):
                del content["error_message"]
                response = ToolMessage(content=json.dumps(content) + "." + self.tool_resp_instr, 
                                       tool_call_id=tool_call_id
                                    )
            else:
                raise Exception(f"Unknown ToolCall Mismatch. Executed Tool is - {tool_call_name}.")

            print("\n ToolMessage Wrapper: ", response)

            return(response)

        except Exception as exp:
            error = f"Error inside RegisterationLookupAgentMiddleware.awrap_tool_call - {exp}"
            print(error, flush=True)
            raise Exception(error)

# Agent Tools
class RegisterationLookupAgentFactory:

    def __init__(self):
        pass

    # Get Required MCP Tools
    async def get_tools(self, mcp_client: MCPClient) -> list[StructuredTool]:
        try:
            lookup_tools = []

            # Set Lookup Tool Filter
            lookup_tool_filter = {'list_users', 'lookup_user'}
            mcp_tools = await mcp_client.get_mcp_tools()
            for tool in mcp_tools:
                if (tool.name in lookup_tool_filter):
                    lookup_tools.append(tool)

            return(lookup_tools)

        except Exception as exp:
            error = f"Error inside RegisterationLookupAgentFactory.get_tools - {exp}"
            print(error, flush=True)
            raise Exception(error)

    async def initialize_lookup_agent(self, model: ChatOllama, tools: list[StructuredTool], 
                                      system_instr: str, tool_inst: str):
        try:

            agent = create_agent(
                model = model,
                tools = tools,
                middleware = [RegisterationLookupAgentMiddleware(tool_inst)],
                system_prompt=SystemMessage(content=system_instr)
            )

            return(agent)

        except Exception as exp:
            error = f"Error inside RegisterationLookupAgentFactory.initialize_lookup_agent - {exp}"
            print(error, flush=True)
            raise Exception(error)

    # Format Agent Response. Index for AIMessage is 1.
    def format_agent_response(self, ai_msg: list[AnyMessage]):
        ai_msg = (ai_msg['messages'][-1]).content
        return(ai_msg.strip())


# Lookup User Registeration
class RegisterationLookup:

    def __init__(self):
        self.reglkp_agent = None
        self.reglkp_resp_instr = None
        self.initialized_status = False
        self.agent_model_name = "llama3.1:8b-instruct-q4_K_M"

        # Init External Classes
        self.mcp_client = MCPClient()

        # Init Internal Classes
        self.reglkp_agent_llm = RegisterationLookupAgentLLM()
        self.reglkp_agent_factory = RegisterationLookupAgentFactory()

    # Initialize Lookup Agent
    @traceable
    async def initialize(self, reglkp_insructions: str, reglkp_resp_insructions: str, lookup_toolcall_instr: str) -> bool:
        try:
            # Get Tool and LLM Instance.
            tools = await self.reglkp_agent_factory.get_tools(self.mcp_client)
            llm_model = self.reglkp_agent_llm.create_reglookup_llm(self.agent_model_name, tools)

            # Instantialte Agent
            self.reglkp_resp_instr = reglkp_resp_insructions        # Lookup Response Instructions.
            self.reglkp_agent = await self.reglkp_agent_factory.initialize_lookup_agent(llm_model, tools, reglkp_insructions, lookup_toolcall_instr)

            self.initialized_status = True

            return(self.initialized_status)

        except Exception as exp:
            error = f"Error inside RegisterationLookup.lookup - {exp}"
            print(error, flush=True)
            raise Exception(error)

    # Get Formated Data Struct for EntryPoint 'lookup' function
    # To Enable it as a Tool Call option in Client App LLMs
    async def get_lookup_tool_spec(self, func_name: str) -> dict:
        try:
            if (func_name == "lookup"):
                description = """ Tool provides ability to - 
                                    1. Get List of Registered Users.
                                    2. Lookup a specific Registered User's Information.

                            Args:
                                user_input: Input Dictionary that will contain User Prompt.                      
                                            'user_prompt' - SHOULD BE User Request as entered in the chat in Plain Text. 
                                                            DO NOT Add Any Extra Commentary.

                            Returns: 
                                response: An output Message containing Results of Lookup Request.

                            """
                tool_spec = {
                    "name": "lookup",
                    "description": description,
                    "args_schema": RegisterationLookupInput
                }

            return(tool_spec)
        
        except Exception as exp:
            error = f"Error inside RegisterationLookup.get_lookup_tool_spec - {exp}"
            print(error, flush=True)
            raise Exception(error)

    # Entry Point Function to Lookup User Registeration
    @traceable
    async def lookup(self, user_prompt: str) -> str:
        try:
            print("Inside RegLookup-1: ", user_prompt, flush=True)

            # Run Agent Inve Async, Since Langchain-mcp-adapter invokes 
            # MCP server Async - If not we get StructuredTool Error
            user_prompt = {"messages": HumanMessage(content=user_prompt)}
            response = await self.reglkp_agent.ainvoke(user_prompt)       
            agent_response = self.reglkp_agent_factory.format_agent_response(response)

            #print("\nResponse RegLookup-2: ", response, flush=True)
            #print("\nResponse RegLookup-3: ", agent_response, flush=True)

            # Appending Resp Instr to Lookup Agent Respinse
            # For RunVent Agent to take Action on this Response
            return(agent_response + self.reglkp_resp_instr)

        except Exception as exp:
            error = f"Error inside RegisterationLookup.lookup - {exp}"
            print(error, flush=True)
            raise Exception(error)
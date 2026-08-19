# Class Late Binding Annotaion
from __future__ import annotations 

# Import Python Native Lib
import os, sys
import json
from dotenv import load_dotenv

from typing import Callable, Awaitable, Any

# Import Langchain Lib
from langchain_mcp_adapters.client import MultiServerMCPClient

# Load Env Variables
load_dotenv()


# Local MCP Client
class MCPClient:

    def __init__(self):
        mcp_url = os.getenv('mcp_runevent_url')

        self.mcp_client = MultiServerMCPClient(
            {
                "mcp_runevent": {
                    "url": mcp_url,
                    "transport": "streamable_http"
                }
            }
        )
        self.mcp_tool_list = None


    # Get Tools from the Server
    async def get_mcp_tools(self):
        try:
            self.mcp_tool_list = await self.mcp_client.get_tools()

            return(self.mcp_tool_list)
        
        except Exception as exp:
            error = f"Error getting in List of MCP Tools - {exp}"
            print(error)
            return(None)

    # Get Tools from the Server
    async def execute(self, tool_name: str, tool_args: dict) -> str:
        tool = None
        for tool in self.mcp_tool_list:
            if (tool.name == tool_name):
                resp = await tool.ainvoke(tool_args)
                resp = resp[0]['text']
                break
            else:
                resp = "Tool Not Found"

        return(resp)

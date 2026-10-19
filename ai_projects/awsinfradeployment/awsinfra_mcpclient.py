# MCP Client

# Class Late Binding Annotaion
from __future__ import annotations 

# Import Python Native Lib
import os, sys
import json
from dotenv import load_dotenv

# Import Langchain Lib
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

# Load Env Vars
load_dotenv()

# Init Langsmith Tracing
#os.environ["LANGSMITH_TRACING"] = "true"
#os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
#os.environ["LANGSMITH_PROJECT"] = os.getenv("runevent_monitor_app")


# Local MCP Client
class MCPClient:

    def __init__(self):
        mcp_url = os.getenv('mcp_awsinfra_url')

        self.mcp_client = MultiServerMCPClient(
            {
                "mcp_awsinfra": {
                    "url": mcp_url,
                    "transport": "streamable_http",
                    "timeout": 1800,
                    "sse_read_timeout": 1800
                }
            }
        )
        self.mcp_tool_list = None


    # Get Tools from the Server
    async def get_mcp_tools(self) -> list[StructuredTool]:
        try:
            self.mcp_tool_list = await self.mcp_client.get_tools()

            return(self.mcp_tool_list)
        
        except Exception as exp:
            error = f"Error inside MCPClient.get_mcp_tools: {exp}"
            print(error, flush=True)
            raise Exception(error)  

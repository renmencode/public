# EC2 Deployment Sub-Agent for AWS Infra.

# Class Late Binding Annotaion
from __future__ import annotations 

# Import Python Native Lib
import os, sys
import json
import asyncio
import uuid as uuid
from dotenv import load_dotenv

# Import DeepAgent Modules
from deepagents.backends import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig
from langchain.agents.middleware import ToolRetryMiddleware

# Import Extrnal Modules
from awsinfra_mcpclient import MCPClient
from awsinfra_deploysubagent_mdlware import DeploySubAgentMiddleware   # Instantiated using class variable 'global()'

# Load Env Vars
load_dotenv()

# loading File Reference
awsinfra_artifactdeploy_subagent_file = "C:\\RanjithC\\AIProjects\\PromptEngg\\langgraph\\data\\awsinfra\\awsinfra_deploysubagent_metadata.json"


# Artifacts Deploy SubAgent
class ArtifactDeploySubAgent:

    def __init__(self):
        self.subagent_metadata = None
        # Load & Retrieve SubAgent Metadata
        with open(awsinfra_artifactdeploy_subagent_file, 'r') as instr:
            self.subagent_metadata = json.load(instr)

        # Init External Class
        self.mcp_client = MCPClient()

    # Reformatting SubAgent Tools Metadata
    # Replace String List with StrucTool list
    async def reformat_subagent_tools(self, metadata: dict) -> list:
        try:
            metadata_structool_list = []
            metadata_toolname_list = metadata["tools"]
            mcp_tool_list = await self.get_tools()

            # Loop thru every Tool(String List) in Subagent Metadata
            # Compare its presence in MCP List (StrucTool liss)
            tool = None
            for tool in metadata_toolname_list:
                tmp_tool = None
                for tmp_tool in mcp_tool_list:
                    if (tool == tmp_tool.name):
                        metadata_structool_list.append(tmp_tool)
                    else:
                        pass

            #print("\nSubAgent Tool Reformat: ", metadata_structool_list, flush=True)

            return(metadata_structool_list)
        
        except Exception as exp:
            error = f"Error inside ArtifactDeploySubAgent.reformat_subagent_tools: {exp}"
            print(error, flush=True)
            raise Exception(error)

    # Reformatting SubAgent Middleware Metadata
    # Reformattign Skip the Default Middlewares of Langchain
    # For Defaults we need to write Instantiation Code
    async def reformat_subagent_middleware(self, metadata: dict) -> list:
        try:
            mdw_instance_list = []
            golden_template_location = os.getenv("awsinfra_artifact_templates")

            mdw_list = metadata["middleware"]
            for mdw in mdw_list:
                mdw_class_obj = None
                mdw_class = mdw["class_name"]
                if (mdw_class == "FilesystemMiddleware"):
                    file_backend = FilesystemBackend(root_dir=golden_template_location, virtual_mode=True)
                    mdw_class_obj = FilesystemMiddleware(
                        backend = file_backend,
                        tools = ["ls", "read_file"]
                    )
                    mdw_instance_list.append(mdw_class_obj)    # mdw_class_obj os already an instance

                elif (mdw_class == "HumanInTheLoopMiddleware"):
                    mdw_class_args = mdw.get("args", {})
                    interrupt_on = mdw_class_args.get("interrupt_on", {})

                    interrupt_on_dict = {}
                    for key, value in interrupt_on.items():
                        interrupt_on_dict[key]=InterruptOnConfig(
                            description=value,
                            allowed_decisions=["edit"]
                        )

                    mdw_class_obj = HumanInTheLoopMiddleware(
                        interrupt_on = interrupt_on_dict
                    )
                    mdw_instance_list.append(mdw_class_obj)

                else:
                    mdw_class_args = mdw.get("args", {})

                    mdw_class_obj = globals()[mdw_class]
                    mdw_instance_list.append(mdw_class_obj(**mdw_class_args))  # mdw_class_obj need to be instantiated

            #print("\nSubAgent Middleware Reformat: ", mdw_instance_list, flush=True)

            return(mdw_instance_list)

        except Exception as exp:
            error = f"Error inside ArtifactDeploySubAgent.reformat_subagent_middleware: {exp}"
            print(error, flush=True)
            raise Exception(error)

    # Get List of SubAgent Metadata
    async def get_subagents(self) -> list[dict]:
        try:
            metadata_list = []

            # Read Sub Agents Metadata and Replace Tools (String List)
            # With Tools Object (StructuredTool List)
            metadata_keys = self.subagent_metadata.keys()
            for metadata_key in metadata_keys:
                metadata = self.subagent_metadata[metadata_key]

                # Replace Tool(String List) with Tools(StrucTool List)
                if ("tools" in metadata):
                    structool_list = await self.reformat_subagent_tools(metadata)
                    metadata["tools"] = structool_list

                # Replace Middleware(String List) with Middleware Instance List
                if ("middleware" in metadata):
                    middleware_list = await self.reformat_subagent_middleware(metadata)
                    metadata["middleware"] = middleware_list

                metadata_list.append(metadata)

            #print(f"\nSubAgent List {metadata_keys}: ", metadata_list, flush=True)
                
            return(metadata_list)

        except Exception as exp:
            error = f"Error inside ArtifactDeploySubAgent.get_subagents: {exp}"
            print(error, flush=True)
            raise Exception(error)

    # Get SubAgent
    async def get_tools(self) -> list:
        try:
            mcp_tools = await self.mcp_client.get_mcp_tools()

            print("\nMCP Tools: ", mcp_tools, flush=True)

            return(mcp_tools)

        except Exception as exp:
            error = f"Error inside ArtifactDeploySubAgent.get_tools: {exp}"
            print(error, flush=True)
            raise Exception(error)

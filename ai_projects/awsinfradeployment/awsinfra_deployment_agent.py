# Primary Agent for AWS Infra Deloyment.

# Class Late Binding Annotaion
from __future__ import annotations 

# Import Python Native Lib
import os, sys
import json
import asyncio
import uuid as uuid
from dotenv import load_dotenv

# Import Langchain Packages
from deepagents import create_deep_agent, DeepAgentState, HarnessProfile, GeneralPurposeSubagentProfile, register_harness_profile
from langchain_core.tools import StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, AnyMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

# Import OpenAI packages
from langchain_openai import ChatOpenAI

# Add Custom Modules
from awsinfra_userintent import UserIntent
from awsinfra_deployagent_mdlware import DeployAgentMiddleware
from awsinfra_artifactdeploy_subagent import ArtifactDeploySubAgent

# Load Env Vars
load_dotenv()

# Loading Model Names
userintent_model_name = os.getenv("awsinfra_userintentcheck_model")
deployment_model_name = os.getenv("awsinfra_deployment_model")

# loading File Reference
awsinfra_agent_instructions_file = "C:\\RanjithC\\AIProjects\\PromptEngg\\langgraph\\data\\awsinfra\\awsinfra_agent_instructions.json"

# Removing General Purpose SubAgent
# From Deep Agent Configuration.
harness_profle = HarnessProfile(
    general_purpose_subagent = GeneralPurposeSubagentProfile(
        enabled = False
    )
)
register_harness_profile("openai", harness_profle)
register_harness_profile("anthropic", harness_profle)


# DeepAgent State Schema
class AWSInfraAgentState(DeepAgentState):
    baked_launch_template: str | None
    artifact_deploy_output: dict | None

# Agent LLM Class
class AgentLLM:

    def __init__(self):
        pass

    # Instantiate Claude Sonnet LLM
    def instantiate(self, model_name: str) -> ChatOpenAI:
        try:

            llm_model = ChatOpenAI(
                model=model_name,
                temperature=0.0,
                max_retries=2
            )

            return(llm_model)
        
        except Exception as exp:
            error = f"Error inside AgentLLM.instantiate_llm: {exp}"
            print(error, flush=True)
            raise Exception(error)

# Agent Factory Class
class AWSInfraAgentFactory:

    def __init__(self, middleware_instr: dict):
        self.middleware_instr = middleware_instr

    # Set Config Object
    def set_state_config(self):
        thread_id = uuid.uuid4()
        self.state_config = {"configurable": {"thread_id": thread_id}}

    # Get Config Object
    def get_state_config(self) -> dict:
        return(self.state_config)
    
    # Get tools for Agent
    async def get_agent_tools(self, user_intent: UserIntent) -> list[StructuredTool]:
        try:
            tools = []

            # Set Local tools
            # return_direct - 'False' to direct ToolCall Resonse to LLM
            tool_spec = await user_intent.get_tool_spec("check_intent")
            local_tool = [StructuredTool.from_function(
                name = tool_spec['name'],
                description = tool_spec['description'],
                args_schema = tool_spec['args_schema'],
                return_direct = False, 
                response_format = "content",
                coroutine = user_intent.check_intent
            )]
            tools = local_tool

            return(tools)

        except Exception as exp:
            error = f"Error inside AWSInfraAgentFactory.get_agent_tools: {exp}"
            print(error, flush=True)
            raise Exception(error)  

    # Create InfraDeployment Agent
    async def create_deployment_agent(self, model: ChatOpenAI, agent_tools: list[StructuredTool],
                                    subagents: list[dict], system_instr: str) -> CompiledStateGraph:
        try:
            checkpoint = InMemorySaver()
            self.set_state_config()

            agent = create_deep_agent(
                model=model,
                tools=agent_tools,
                middleware=[DeployAgentMiddleware(self.middleware_instr)],
                system_prompt=SystemMessage(content=system_instr),
                checkpointer=checkpoint,
                subagents=subagents,
                state_schema=AWSInfraAgentState
            )

            return(agent)

        except Exception as exp:
            error = f"Error inside AWSInfraAgentFactory.create_deployment_agent: {exp}"
            print(error, flush=True)
            raise Exception(error)


# AWS Infra Agent Class
class AWSInfraAgent:

    def __init__(self):

        self.instr_config = None
        self.interrupt_flag = False

        # Load & Retrieve Agent Instructions
        with open(awsinfra_agent_instructions_file, 'r') as instr:
            self.instr_config = json.load(instr)

        # Init Internal Classes
        self.deployment_agent = None
        self.deployagent_mdlware_instr = {
            "check_intent": self.instr_config["check_intent_mdlware_instruction"],
            "agent_fileops_skip_instruction": self.instr_config["agent_native_fileops_skip_instruction"]
        }

        self.agent_llm = AgentLLM()
        self.agent_factory = AWSInfraAgentFactory(self.deployagent_mdlware_instr)

        # Init External Classes
        self.user_intent = UserIntent(self.instr_config["userintent_instruction"], 
                                      self.instr_config["checkintent_tool_docstring"]
                                    )
        self.artifact_deploy_subagent = ArtifactDeploySubAgent()

    # Initialize Agent
    async def initialize(self):
        try:
            # Initialize User Intent Class
            self.user_intent.set_deployable_artifacts(self.instr_config["deployable_list"])
            self.user_intent.instantiate_llm(userintent_model_name)

            # Initialize AgentLLM
            agent_llm = self.agent_llm.instantiate(deployment_model_name)

            # Get Agent Tools, Get SubAgents
            # Initialize the Orchestrator Agent
            agent_instruction = self.instr_config["agent_instruction"]
            agent_tools = await self.agent_factory.get_agent_tools(self.user_intent)
            agent_subagents = await self.artifact_deploy_subagent.get_subagents()
            self.deployment_agent = await self.agent_factory.create_deployment_agent(agent_llm, agent_tools, agent_subagents, agent_instruction)

            return(True)

        except Exception as exp:
            error = f"Error inside AWSInfraAgent.initialize: {exp}"
            print(error, flush=True)
            raise Exception(error)

    # Run Agent
    async def run(self, user_prompt: str | dict):
        try:
            config = self.agent_factory.get_state_config()

            if (self.interrupt_flag == False):
                resp_message = await self.deployment_agent.ainvoke({"messages": [HumanMessage(content=user_prompt)]}, config=config)
                print("\nAI Response-1: ", resp_message, flush=True)

                # Interrupt for Human Engagement (HITL)
                if ('__interrupt__' in resp_message):
                    self.interrupt_flag = True
                    interrupt_resp = (resp_message["__interrupt__"])[0]
                    interrupt_payload = interrupt_resp.value["action_requests"]
                    interrupt_name = (interrupt_payload[0])["name"]
                    interrupt_desc = (interrupt_payload[0])["description"]
                    
                    resp_message = {
                        "interrupt_flag": True,
                        "interrupt_name": interrupt_name,
                        "message": interrupt_desc
                    }
                    print("\nAI INTERRUT Response-2: ", resp_message, flush=True)

                else:
                    resp_message = {
                        "message": (resp_message["messages"][-1]).content
                    }

            else:
                self.interrupt_flag = False
                resp_message = await self.deployment_agent.ainvoke(Command(resume=user_prompt), config=config)
                print("\nAI Response-3: ", resp_message, flush=True)
                resp_message = {
                    "message": (resp_message["messages"][-1]).content
                }
            
            return(resp_message)
        
        except Exception as exp:
            error = f"Error inside AWSInfraAgent.run: {exp}"
            print(error, flush=True)
            raise Exception(error)  



        
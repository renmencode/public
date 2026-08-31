# Notes:
# User Registration Sub-Agent. This is invoked from Run Event Agent.
#
# Functionality:
# 1. Agent Technology - LangGraph
# 2. LLM Used - OpenAI GPT4.1
# 3. Input : Exact Prompt that User input. 
# 4. Output: Rgistration Success or Failuer Message with Registration ID.
# 5. Tools: Publised as MCP Services.
# 6. Point of Entry - 'initialize' and then 'register'.
# 7. System Instruction: runningevent_agent_instructions.json #user_registeration.
#
# Proces Flow:
# 1. LLM Node is the prmary Entry point of the Graph
# 2. Process Node are mix of Custom Nodes and Tool Nodes.
# 3. ToolNodes Execution is via UserRegisterationTools.tool_executor
# 4. The above approach help to manipulate the ToolCall & Graph State, before and after Tool Execution.
# 5. At End of Graph Execution, the Context is shrunk to Initialization State, to manage Token Cost and Context Overrun.
# 6. Registration pocess is setup as a 'Routing Slip' pattersn where the Task or Steps are provided as JIT Context.
# 7. Each step of the Route / Turn is provided in this file - user_registeration_task_context.csv
# 8. The file gets loaded as part of the initialization of the this Program / Module. 
# 9. They have four columns - Task Name, Tool Name, Request Context, Response Context.
# 10. For each turn, system reads task, context and execute then in teh provided order. 
# 11. Incase of Execption the flow is cut short and appropriate Error is sent to Agent layer.
# 12. Incae of Success, appropriate messsage is formatted and sent to Agent Layer. 
# 13. At end of execution Context is clean-up and kept ready for the next request from Agent.  


# Class Late Binding Annotaion
from __future__ import annotations 

# Import Python Native Lib
import os, sys
import json
import numpy as np
import uuid as uuid
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict, Literal

# Import Pandas
import pandas as pd

# Import LangGraph & LangChain packages
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import InjectedState
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, AnyMessage, RemoveMessage
from langsmith import traceable

# Import Langchain AI Models
from langchain_openai import ChatOpenAI

# Custom Classes
from runevent_mcpclient import MCPClient
from runevent_sentiment import SentimentAnalysis

userreg_task_ctx_file = "C:\\RanjithC\\AIProjects\\PromptEngg\\langgraph\\data\\user_registeration_task_context.csv"

# Load Env Variables
load_dotenv()

# Init Langsmith Tracing
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("runevent_monitor_app")


# User Info
class UserInfo(BaseModel):
    first_name: str
    last_name: str
    user_age: int
    user_email: str
    user_phone: int

# LangGraph State
class State(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]            
    out_messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)       # Pruned "out_messages" gets collected for every Graph Run

# Input Basemodel is created as a wrapper for all Tool Func Input Args. 
# Vars in the BaseModel will be specificed as Args in Tool Func.
# If any Var is a Dict, that need to defined inside this wrapper BaseModel.
class UserRegRunInput(BaseModel):
    user_prompt: str = Field(description="User Prompt", default_factory=str)


# Agent LLM Class
class UserRegisterationLLM:

    def __init__(self):
        self.op_user_register_llm_tools = None              # OpenAI LLM with Tools

    # Instantiate OpenAI LLM
    def create_user_registeration_llm(self, model_name: str, tools):

        op_event_register_llm = ChatOpenAI(
            model = model_name, 
            temperature = 0.0, 
            max_retries = 2
        )
        # Bind with Tools.
        self.op_user_register_llm_tools = op_event_register_llm.bind_tools(tools)


    # Prefix AI or AI (Tool Call) to Conversation Content of AIMessage
    def prefix_ai_to_content(self, msg: AIMessage) -> AIMessage:
        if msg.content == '':
            msg.content = "AI (tool call):"
        else:
            if msg.content.startswith("AI:"):
                pass
            else:
                msg.content = "AI: " + msg.content
        return(msg)

    # Invoke OpenAI Chat Node
    def chat_llm_node(self, state: State):
        try:

            ai_message = self.op_user_register_llm_tools.invoke(state.messages)
            return({"messages": [self.prefix_ai_to_content(ai_message)]})   # Prepend AI to Content Value.
        
        except Exception as exp:
            error = f"Exception inside UserRegisterationLLM.chat_llm_node - {exp}"
            print(error, flush=True)
            raise Exception(error)


# Initialize LangGraph's StateGraph
class UserRegisterationGraph:

    def __init__(self):
        self.graph = None
        self.graph_config = None

    # Build State Graph
    def build(self, agent_llm: UserRegisterationLLM, agent_tools: UserRegisterationTools) -> StateGraph:
        try:

            # graph_builder = StateGraph(state_schema=State, output_schema=OutputState)

            # Build Langgraph Flows
            # Output_Schema (attibute) rolled up inside State. 
            graph_builder = StateGraph(state_schema=State)

            # Add Nodes
            graph_builder.add_node("llm_node", agent_llm.chat_llm_node)
            graph_builder.add_node("user_tools", agent_tools.tools_executor)
            graph_builder.add_node("end_graph", self.prune_graph_output)

            #Add Edges
            graph_builder.add_edge(START, "llm_node")
            graph_builder.add_conditional_edges("llm_node", self.tools_condition)
            graph_builder.add_edge("user_tools", "llm_node")
            graph_builder.add_edge("end_graph", END)

            # Add Checkpoints to Memory
            memory= MemorySaver()

            # Build the Graph
            self.graph = graph_builder.compile(checkpointer=memory, name="event_registration_agent")

            return(self.graph)
        
        except Exception as exp:
            error = f"Exception inside UserRegisterationGraph.build - {exp}"
            print(error, flush=True)
            raise Exception(error)

    # Override langgraph default 'tools_condition'
    def tools_condition(self, state: State) -> Literal["user_tools", "end_graph"]:
        try:
            # print("\nBBB: ", state.messages[-1])
            msg =  state.messages[-1]   
            if (isinstance(msg, AIMessage)) and (msg.additional_kwargs.get("tool_calls") is not None):
                tool_call = msg.additional_kwargs.get("tool_calls")         # Get "tool_calls" Structure
                func_name = tool_call[0].get("function").get("name")        # Get Function Name from tool_calls Struct
                if (func_name == "validate_user"):
                    return("user_tools")
                elif (func_name == "register_user"):
                    return("user_tools")
                elif (func_name == "notify_user"):
                    return("user_tools")
                else:
                    #return(END)
                    return("end_graph")                 # Default Function Returned when routed to END 
            else:
                #return(END)
                return("end_graph")                     # Default Function Returned when routed to END
            
        except Exception as exp:
            error = f"Exception inside UserRegisterationGraph.tools_condition - {exp}"
            print(error, flush=True)
            raise Exception(error)

    # Set Graph Config        
    def set_graph_config(self):
        thread_id = uuid.uuid4()                        # Initialise Agent Thread
        self.graph_config = {"configurable": {"thread_id": thread_id}}

    # Get Graph & Config
    def get_graph_config(self) -> tuple:
        return(self.graph, self.graph_config) 

    # Filter / Remove ToolMessages & AIMessage with Funcation Call.
    def filter_valid_message(self, msg: str) -> bool:
        if (isinstance(msg, ToolMessage)):
            return False
        elif (isinstance(msg, AIMessage)) and (msg.additional_kwargs.get("tool_calls") is not None):
            return False
        else:
            return True

    # Filter / Remove SysMessages, HumanMessage, AIMessage.
    def filter_invalid_message(self, msg: str) -> bool:
        if (isinstance(msg, ToolMessage)):
            return True
        elif (isinstance(msg, AIMessage)) and (msg.additional_kwargs.get("tool_calls") is not None):
            return True
        else:
            return False

    # This Function is added as an intrcept before routing to END from LLM
    # Used to remove unwanted Messages from the Interal State Object
    def prune_graph_output(self, state: State) -> dict:
        try:
            # Filter & Collect Good Messages - Sys, HM, AI (without ToolCall)
            valid_msg = list(filter(self.filter_valid_message, state.messages))

            # Filter & Collect Invalid Messages - ToolMessage, AI (tool call)
            invalid_msg = list(filter(self.filter_invalid_message, state.messages))

            # Setting Remove Pointer in State Obj
            remove_msg = []
            for bad_msg in invalid_msg:
                remove_msg.append(RemoveMessage(id=bad_msg.id))

            # Formatting the 'out_messags' for State Obj
            out_msg = []
            for msg in valid_msg:
                if (isinstance(msg, AIMessage)):
                    out_msg = out_msg + [AIMessage(content=msg.content, id=msg.id)]
                else:
                    out_msg = out_msg + [msg]

            return({
                "messages": remove_msg,         # 'messages' gets updated / removed based on Msg.id inside State 
                "out_messages": out_msg         # 'out_messages' gets created inside State & updated in OutputState
            })                                  # since this is the End Node of Graph.

        except Exception as exp:
            error = f"Exception inside UserRegisterationGraph.prune_graph_output - {exp}"
            print(error, flush=True)
            raise Exception(error)

    # Final Step in Service Execution flow.
    # Function is called OUTSIDE OF GRAPH Execution after it Ends.
    # Used to Remove All Messages except first 3 - Initial Sys, HM Selutation & its AI Response 
    def cleanup_graph_state(self):
        try:

            snapshots = self.graph.get_state(self.graph_config)   
            state_obj = snapshots.values        # .values Return a Python 'Dict' and Not Pydantic.
 
            remove_msg = []
            for state_msg in state_obj["messages"][3:]:
                remove_msg.append(RemoveMessage(id=state_msg.id))

            self.graph.update_state(
                self.graph_config,
                {
                    "messages": remove_msg
                }
            )

        except Exception as exp:
            error = f"Exception inside UserRegisterationGraph.cleanup_graph_state - {exp}"
            print(error, flush=True)
            raise Exception(error)

    # Invoke Graph
    async def ainvoke(self, user_prompt: list[AnyMessage]) -> AIMessage:
        try:
            ai_response = await self.graph.ainvoke({"messages": user_prompt}, self.graph_config)
            return(ai_response)
        except Exception as exp:
            error = f"Exception inside UserRegisterationGraph.ainvoke - {exp}"
            print(error, flush=True)
            raise Exception(error)


# User Reg Agent Factory
class UserRegisterationUtils:

    def __init__(self, userreg_sentiment: SentimentAnalysis):
        self.userreg_sentiment = userreg_sentiment


    # Set Tool Resp Context Pandas object after File Read
    def set_task_resp_ctx(self, userreg_task_resp_ctx: pd):
        self.userreg_task_resp_ctx = userreg_task_resp_ctx

    # Get the ToolCallId from tool_call_Id List - NOT USED
    def get_tool_call_Id(self, tool_call_list: list, func_name: str) -> str:
        
        func_id = ""
        for tool in tool_call_list:
            if tool.get("function").get("name") == func_name:
                func_id = tool.get("id")
                break
        
        return(func_id)

    # Get Task Transition Response
    def get_task_resp_context(self, tool_name: str, tool_resp: str) -> str:
        try:
            temp_ctx = self.userreg_task_resp_ctx
            tool_resp_ctx = list(temp_ctx[temp_ctx["Tool"] == tool_name]["ResponseCtx"])[0]

            tool_resp = tool_resp.strip()
            tool_resp_sentiment = self.userreg_sentiment.check_sentiment(tool_resp)   # Check the Sentiment of Tool Response

            if (tool_resp_sentiment == "POSITIVE"):
                tool_resp_ctx = tool_resp_ctx.split("Failure:")[0].strip()     # Strip-Out Failure Section.
                tool_resp_ctx = tool_resp_ctx.split("Success:")[1].strip()     # Strip-Out 'Success:' from remaineg context
            elif (tool_resp_sentiment == "NEGATIVE"):
                tool_resp_ctx = tool_resp_ctx.split("Failure:")[1].strip()     # Strip-Out all Text Prior to 'Failure:' Section.
            else:
                tool_resp_ctx = ""

            return(tool_resp_ctx)
        
        except Exception as exp:
            error = f"Error inside UserRegisterationUtils.get_task_resp_context - {exp}"
            print(error, flush=True)
            raise Exception(error)


# Create User Registeration Agent
class UserRegisterationTools:

    def __init__(self, mcp_client: MCPClient, agent_utils: UserRegisterationUtils):
        self.agent_utils = agent_utils
        self.mcp_client = mcp_client

    # Agent Tool Executor. This is replacement for Tool Node
    # Application has the responsiblity to route to appropriate Tool.
    # Tool Executor is called whenever LLM need to make a Tool Call used Tools Registered in LLM
    # Tool Executor is triggered by Tool_Condition Response, that LLM invokes with appropriate Tool Name.
    async def tools_executor(self, state: Annotated[dict, InjectedState]) -> Command:

        func_name = None
        func_args = None

        try:
            tool_message = state.messages[-1]
            tool_calls = tool_message.additional_kwargs.get("tool_calls")        # Get "tool_calls" Structure

            tool_call_id = tool_calls[0].get("id")                               # Get Tool Call ID
            func_name = tool_calls[0].get("function").get("name")                # Get Func Name
            func_args = tool_calls[0].get("function").get("arguments")           # Get Tool Call Args 

            if isinstance(func_args, str):                                       # Model Tool Args as JSON.
                args_json = json.loads(func_args)                                # Even if they have only One field.
            else:
                args_json = func_args                                            # func_args expcted to be JSON

            response = await self.mcp_client.execute(func_name, args_json)

            task_resp_ctx = self.agent_utils.get_task_resp_context(func_name, response)    # Get Tool Response Context from response.CSV 
            response = response + task_resp_ctx                              # Concat both the Response

            return(Command(update={
                "messages": [ToolMessage(content=response, tool_call_id=tool_call_id)]
            }))
        
        except Exception as exp:
            error = f"Error inside UserRegisterationTools.tools_executor - {exp}"
            print(error, flush=True)

            tool_error = f"Error in ToolCall: {func_name} with Arguments: {func_args}"    
            return(Command(update={
                "messages": [ToolMessage(content=tool_error, tool_call_id=tool_call_id, status="error")]
            }))

    # Launch User Registration Agent.
    async def launch(self, agent_graph: UserRegisterationGraph, system_instruction: str) -> AIMessage:
        try:
            # Agent Instructions
            system_prompt_content = system_instruction
            
            input_prompt = [SystemMessage(content=system_prompt_content), 
                            HumanMessage(content="Hello!")
            ]

            resp_message = await agent_graph.ainvoke(input_prompt)

            return(resp_message)

        except Exception as exp:
            error = f"Exception inside UserRegisterationTools.launch - {exp}"
            print(error, flush=True)
            raise Exception(error)

    # Get MCP Tools Relevent to this Agent
    # Tools are - 'validate_user', 'register_user' from MCP Server
    async def set_agent_tools(self) -> list[StructuredTool]:
        try:
            # Get Tool from MCP Server and 
            # Filter based on ToolList(Only Include)
            tool_list = {'validate_user', 'register_user', "notify_user"}
            mcp_tools = await self.mcp_client.get_mcp_tools()
            tools = [
                tmp_tool
                for tmp_tool in mcp_tools
                if tmp_tool.name in tool_list
            ]

            return(tools)

        except Exception as exp:
            error = f"Error inside UserRegisterationTools.set_agent_tools - {exp}"
            print(error, flush=True)
            raise Exception(error)


# User Registration LangGraph Agent. 
class UserRegisteration:

    def __init__(self):
        self.initialized_status = False
        self.pd_userreg_task_resp_ctx = None
        self.agent_model_name = "gpt-4.1"

        # Init External Classes
        self.mcp_client = MCPClient()
        self.userreg_sentiment = SentimentAnalysis()

        # Init Internal Classes
        self.userreg_agent_llm = UserRegisterationLLM()
        self.userreg_agent_utils = UserRegisterationUtils(self.userreg_sentiment)
        self.userreg_agent_tools = UserRegisterationTools(self.mcp_client, self.userreg_agent_utils)
        self.userreg_agent_graph = UserRegisterationGraph()

    # Initialise Agent Graph
    @traceable
    async def initialize(self, userreg_instruction: str) -> bool:
        try:
            self.userreg_agent_graph.set_graph_config()                      # Initialise Agent Thread

            # Load Pandas Object Agent Util
            self.pd_userreg_task_resp_ctx = pd.read_csv(userreg_task_ctx_file, sep="|")
            self.userreg_agent_utils.set_task_resp_ctx(self.pd_userreg_task_resp_ctx)

            # Instantiate MCPClient
            agent_tools = await self.userreg_agent_tools.set_agent_tools()

            # Instantiate LLM
            self.userreg_agent_llm.create_user_registeration_llm(self.agent_model_name, agent_tools)

            # Build State Graph
            self.userreg_agent_graph.build(self.userreg_agent_llm, self.userreg_agent_tools)

            # Launch Agent
            await self.userreg_agent_tools.launch(self.userreg_agent_graph, userreg_instruction)

            self.initialized_status = True

            return(self.initialized_status)
        
        except Exception as exp:
            error = f"Exception inside UserRegisteration.initialize - {exp}"
            print(error, flush=True)
            raise Exception(error)

    # Get Initialised Status
    async def isInitialized(self):
        return(self.initialized_status)

    # Get Formated Data Struct for EntryPoint 'register' function
    # To Enable it as a Tool Call option in Client App LLMs
    async def get_register_tool_spec(self, func_name: str) -> dict:
        try:
            if (func_name == 'register'):
                description = """ Tool provides ability to Register User Information. 
                                
                            Args:
                                user_prompt: SHOULD BE User Information as entered in Plain Text. 
                                            DO NOT Add Any Extra Commentary.

                            Returns: 
                                response: An Output Message on Success or Failure of Registration.

                            """

                tool_spec = {
                    "name": "register",
                    "description": description,
                    "args_schema": UserRegRunInput
                }

            return(tool_spec)

        except Exception as exp:
            error = f"Error inside UserRegisteration.get_tool_spec - {exp}"
            print(error, flush=True)
            raise Exception(error)

    # EntryPoint for Registeration Service / Agent
    @traceable
    async def register(self, user_prompt: str) -> str:
        try:
            print("\nInside User Register: ", user_prompt, flush=True)

            final_resp_content = None                   # Final AI Response Content for each Route Turn.

            task_ctx = self.pd_userreg_task_resp_ctx
            task_list = task_ctx["Task"].tolist() 

            # Implement Routng Slip in a loop.
            for task in task_list:
                req_ctx = list(task_ctx[task_ctx["Task"] == task]["RequestCtx"])[0]
                req_ctx = req_ctx.strip()

                run_context = req_ctx + ": " + user_prompt
                response = await self.userreg_agent_graph.ainvoke([HumanMessage(content=run_context)])

                # Check if AI Response is Positive or Negative
                final_resp_content = (response["out_messages"][-1]).content
                resp_sentiment = self.userreg_sentiment.check_sentiment(final_resp_content)
                if (resp_sentiment == "NEGATIVE"):
                    break
                else:
                    pass

            # Final Clean-up of Graph State & Resetting Checkpoint
            # Before the Service is exited.
            self.userreg_agent_graph.cleanup_graph_state()

            return(final_resp_content)
        
        except Exception as exp:
            error = f"Error inside UserRegisteration.run - {exp}"
            print(error, flush=True)
            raise Exception(error)


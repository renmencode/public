# Notes:
# Run Event User Registration Agent.
#
# Functionality:
# 1. Agent Technology - LangGraph
# 2. LLM Used - OpenAI GPT4.1, distilbert-base-uncased-finetuned-sst-2-english
# 3. Input : User Interation usin Streamlit Inteface. 
# 4. Output: Chat / Coversational output on User Registeration, Lookup and Listing Users.
# 5. Tools: RAG (FAISS + LLM), 'register' Sub-Agent call & 'lookup' Sub-Agent call 
# 6. Point of Entry - 'initialize' and 'run'.
# 7. System Instruction: runningevent_agent_instructions.json 
#           #runningevent_agent - Primary System Instructions for the Agent
#           #userintent - Instructions and LLM Check the User Intent (of the input Prompt)
#
# Proces Flow:
# 1. This is the Agent for helping User Registertions for the Running Event.
# 2. The Agent is provided with set of FAQs hosted on vector Store - FAISS
# 3. Agent is expected to check RAG first, incase user asked a question.
# 4. If the Answer is not found inside RAG then it is instructed to check if Tools can provide the Answers
# 5. If both the apporach cannot provide any satifactory answer then the Agent say it cannot provide
#    answers for the requested User Prompt / Request. 
# 6. If it is able to fetch the response details, that will be well formatted (as per instructions)
# 7. At the end of the conversation, the over all context should be cleanup (to avoing context 
#    growth and sprawl)
#    - Curretly this approach is being evaluated. 
# 8. One of the technique we have introduced is Prompt Caching. This can be helpful if multile Users 
#    are aksng qustions with the same intent like 'how do i register a user' or 'how do we add a new users'.
# 9. Agent also does Sentiment Checks to understand if the ToolCall has been a Success or a Failure.
#    - Positive or Negative Sentiment is further used a s flow control to proceed with the execution
#      or to cut short the flow and response to the User for further instructions. 
# 10. This Agent is initialized by the Chat Server implemented outside of the Agent Realm.
# 11. Chat Interface (Strealit Clint & Server) acts as the UI for the Agent. 
#    - The Streamlit UI could be replaced by any other UI technologies in future. 
# 12. One of the pending item (TO-DO) is in optimizing Contxt Window.
#   - Based on the Answers to the User Questions like Registerations, Lookup, or List Users
#     once AI answers the request, the context can be summerized or unwnted turns can be removed.    


# Class Late Binding Annotaion
from __future__ import annotations 

# Import Python Native Lib
import os, sys
import json
import asyncio
import numpy as np
import uuid as uuid
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict, Literal
from IPython.display import Image, display

# Import Pandas
import pandas as pd

# Sentence Transformer
from sentence_transformers import CrossEncoder

# Import LangGraph & LangChain packages
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, AnyMessage, RemoveMessage
from langsmith import traceable

# Import Langchain AI Models
from langchain_openai import ChatOpenAI

# Custom Classes
from runevent_mcpclient import MCPClient
from runevent_userintent import UserIntent
from runevent_userregisteration import UserRegisteration
from runevent_lookupregisteration import RegisterationLookup

runningevent_agent_instructions_file = "C:\\RanjithC\\AIProjects\\PromptEngg\\langgraph\\data\\runningevent_agent_instructions.json"

# Load Env Variables
load_dotenv()

# Init Langsmith Tracing
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("runevent_monitor_app")


# LangGraph Message Types
class State(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    out_messages: list[AnyMessage] = Field(default_factory=list)    # Pydantic sets out_messages to new empty list


# Agent LLM Class
class ChatAgentLLM:

    def __init__(self):
        self.ol_intentchecker_llm = None                # Ollama LLM
        self.op_event_register_llm_tools = None         # OpenAI LLM with Tools

    # Instantiate OpenAI LLM
    def create_event_registeration_llm(self, model_name: str, tools: list[StructuredTool]) -> ChatOpenAI:

        op_event_register_llm = ChatOpenAI(
            model=model_name, 
            temperature=0.0, 
            max_retries=2
        )
        # Bind with Tools.
        self.op_event_register_llm_tools = op_event_register_llm.bind_tools(tools)

    # Get event_registeration_llm
    def get_event_registeration_llm(self) -> ChatOpenAI:
        return(self.op_event_register_llm_tools)

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
    def chat_llm_node(self, state: State) -> dict:
        try:
            ai_message = self.op_event_register_llm_tools.invoke(state.messages)   # changed [messages] to .messages
            return({
                "messages": [self.prefix_ai_to_content(ai_message)]       # Prepend AI to Content Value.
            })
        except Exception as exp:
            error = f"Error inside ChatAgentLLM.chat_llm_node: {exp}"
            print(error, flush=True)
            raise Exception(error)


# Initialize LangGraph's StateGraph
class ChatAgentGraph:

    def __init__(self):
        self.graph = None
        self.graph_config = None

    # Build State Graph
    def build(self, agent_llm: ChatAgentLLM, tools: ChatAgentTools) -> StateGraph:
        try:
            # Build Langgraph Flows
            graph_builder = StateGraph(state_schema=State)

            runevent_tools = tools.get_runevent_tools()
            userreg_tools = tools.get_userreg_tools()
            reglkp_tools = tools.get_reglkp_tools()

            # Add Nodes
            #graph_builder.add_node("intent_node", user_intent.check_user_intent)
            graph_builder.add_node("chat_llm_node", agent_llm.chat_llm_node)
            graph_builder.add_node("runevent_tool", ToolNode(runevent_tools, handle_tool_errors=True))
            graph_builder.add_node("userreg_tool", ToolNode(userreg_tools, handle_tool_errors=True))
            graph_builder.add_node("reglkp_tool", ToolNode(reglkp_tools, handle_tool_errors=True))
            graph_builder.add_node("end_graph", self.prune_graph_output)

            #graph_builder.add_edge(START, "intent_node")
            graph_builder.add_edge(START, "chat_llm_node")
            graph_builder.add_conditional_edges("chat_llm_node", self.tools_condition)
            graph_builder.add_edge("runevent_tool", "chat_llm_node")
            graph_builder.add_edge("userreg_tool", "chat_llm_node")
            graph_builder.add_edge("reglkp_tool", "chat_llm_node")
            graph_builder.add_edge("end_graph", END)

            # Add Checkpoints to Memory
            memory= MemorySaver()

            # Build the Graph
            self.graph = graph_builder.compile(checkpointer=memory, name="event_registration_agent")

            # Visualize Graph
            display(Image(self.graph.get_graph().draw_mermaid_png()))

            return(self.graph)

        except Exception as exp:
            error = f"Error inside ChatAgentGraph.build: {exp}"
            print(error, flush=True)
            raise Exception(error)

    # Override langgraph default 'tools_condition'
    def tools_condition(self, state: State) -> Literal["runevent_tool", "userreg_tool", "reglkp_tool", "end_graph"]:
        try:
            #print("\nAAA: ", state.messages[-1])
            msg =  state.messages[-1]                               # Changed [messages] to .messages 
            if (isinstance(msg, AIMessage)) and (msg.additional_kwargs.get("tool_calls") is not None):
                tool_call = msg.additional_kwargs.get("tool_calls")         # Get "tool_calls" Structure
                func_name = tool_call[0].get("function").get("name")        # Get Function Name from tool_calls Struct
                if (func_name == "search_faq"):
                    return("runevent_tool")
                elif (func_name == "lookup"):
                    return("reglkp_tool")
                elif (func_name == "register"):
                    return("userreg_tool")
                else:
                    #return(END)
                    return("end_graph")                 # Default Function Returned when routed to END 
            else:
                #return(END)
                return("end_graph")                     # Default Function Returned when routed to END
            
        except Exception as exp:
            error = f"Error inside ChatAgentGraph.tools_condition: {exp}"
            print(error, flush=True)
            raise Exception(error)

    # Set Graph Config        
    def set_graph_config(self):
        thread_id = uuid.uuid4()                        # Initialise Agent Thread
        self.graph_config = {"configurable": {"thread_id": thread_id}}

    # Get Graph & Config
    def get_graph_config(self) -> tuple:
        return(self.graph, self.graph_config)

    # Manually Updates Messages in State Obj
    def update_state(self, msg_list: list):
        self.graph.update_state(
            self.graph_config,
            {
                "messages": msg_list
            }
        )

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
    
    # Graph Invoke - SHOULD BE Async for MCPTool Call to Work via ToolNode.
    # Regular invocation to LLM can happen via Sync calls.
    async def ainvoke(self, user_prompt: list[AnyMessage]) -> AIMessage:
        try:
            ai_response = await self.graph.ainvoke({"messages": user_prompt}, self.graph_config)
            return(ai_response)
        except Exception as exp:
            error = f"Error inside ChatAgentGraph.ainvoke: {exp}"
            print(error, flush=True)
            raise Exception(error)


# Agent Orchestration
class ChatAgentTools:

    def __init__(self):
        self.cross_encoder = CrossEncoder(
            model_name_or_path = "cross-encoder/ms-marco-MiniLM-L-6-v2",
            cache_folder = os.getenv("hf_cache_folder")
        )
        self.reglkp_tool = None
        self.userreg_tool = None
        self.runevent_tools = None

    # Launch Chat Agent. Not a Tool perse, but Agent Selutation and Launch Instructions.
    async def launch(self, agent_graph: ChatAgentGraph, system_instruction: str) -> dict:
        try:
            # Agent Instructions
            system_prompt_content = system_instruction
            
            input_prompt = [SystemMessage(content=system_prompt_content), 
                            HumanMessage(content="Hello")
            ]

            resp_message = await agent_graph.ainvoke(input_prompt)

            return(resp_message)
        
        except Exception as exp:
            error = f"Error inside ChatAgentTools.launch: {exp}"
            print(error, flush=True)
            raise Exception(error)

    # Check Relevance of Two Inputs <TBD>
    def check_relevance(self, input1: str, input2: str, score_threshold: float) -> bool:
        try:
            # Create a Pair with Two Inputs
            doc_pair = [input1, input2]

            relavence_score = self.cross_encoder.predict(
                inputs = doc_pair,
                device = "cpu",
                convert_to_numpy = True
            )
            print("Relevance Score: ", relavence_score, flush=True)

            if (relavence_score > score_threshold):
                response = True
            else:
                response = False
            
            return(response)
        
        except Exception as exp:
            error = f"Error inside ChatAgentTools.check_relevance: {exp}"
            print(error, flush=True)
            raise Exception(error)

    # Get MCP Tools Relevent to this Agent
    # Tools are - 'search_faq' from MCP Server
    # Register_User Sub-Agent is added as a Tool after filtering
    async def set_agent_tools(self, userreg: UserRegisteration, reglkp: RegisterationLookup, mcp_client: MCPClient) -> list[StructuredTool]:
        try:
            # Get Tool from MCP Server and 
            # Filter based on ToolList(Only Include)
            # tool_list_filter = {'search_faq', 'list_users', 'lookup_user'}
            tool_list_filter = {'search_faq'}
            mcp_tools = await mcp_client.get_mcp_tools()
            self.runevent_tools = [
                tmp_tool
                for tmp_tool in mcp_tools
                if tmp_tool.name in tool_list_filter
            ]

            # Add 'User Registeration' Sub-Agent as StructuredTool
            # Created as List for updating ToolNode in Graph
            tool_spec = await userreg.get_register_tool_spec("register")
            self.userreg_tool = [StructuredTool.from_function(
                name = tool_spec['name'],
                description = tool_spec['description'],
                args_schema = tool_spec['args_schema'],
                return_direct = True,
                response_format = "content",
                coroutine = userreg.register 
            )]

            # Add 'Registeration Lookup' Sub-Agent as Structured Tool
            # Created as List for updating ToolNode in Graph
            tool_spec = await reglkp.get_lookup_tool_spec("lookup")
            self.reglkp_tool = [StructuredTool.from_function(
                name = tool_spec['name'],
                description = tool_spec['description'],
                args_schema = tool_spec['args_schema'],
                return_direct = True,
                response_format = "content",
                coroutine = reglkp.lookup
            )]

            tools = self.runevent_tools + self.userreg_tool + self.reglkp_tool
            return(tools)

        except Exception as exp:
            error = f"Error inside ChatAgentTools.set_agent_tools: {exp}"
            print(error, flush=True)
            raise Exception(error)

    # Get RunEvent Agent RAG Tool
    def get_runevent_tools(self) -> list[StructuredTool]:
        return(self.runevent_tools)

    # Get User Registeration Tool
    def get_userreg_tools(self) -> list[StructuredTool]:
        return(self.userreg_tool)

    # Get Registeration Lookup Tool
    def get_reglkp_tools(self) -> list[StructuredTool]:
        return(self.reglkp_tool)
    

# Create a LangGraph ChatAgent
class ChatAgent:

    def __init__(self):
        # self.exit_flag = False
        self.instr_config = None
        self.userintent_instr = None
        self.runevent_agent_instr = None

        # Init External Classes
        self.mcp_client = MCPClient()
        self.user_intent = UserIntent()

        # Init Internal Classes
        self.agent_llm = ChatAgentLLM()
        self.agent_graph = ChatAgentGraph()
        self.agent_tools = ChatAgentTools()

        # Sub-Agents
        self.userreg_agent = UserRegisteration()
        self.reglkp_agent = RegisterationLookup()

    # Initialise Agent Graph
    @traceable
    async def initialize(self) -> bool:
        try:
            op_model_name = "gpt-4.1"   

            # Load & Retrieve RunEvent Agent Instructions
            with open(runningevent_agent_instructions_file, 'r') as instr:
                self.instr_config = json.load(instr)

            # Map Agent Instructions
            self.userintent_instr = self.instr_config["userintent"]
            self.runevent_agent_instr = self.instr_config["runningevent_agent"]

            # Initialise / Set Graph Config
            self.agent_graph.set_graph_config()

            # Get MCP Tools + Structured Tools to 'bind' with LLM 
            agent_tools = await self.agent_tools.set_agent_tools(self.userreg_agent, self.reglkp_agent, self.mcp_client)
            #print("Agent Tools: ", agent_tools, flush=True)

            # Instantiate Intent Checker LLM
            self.user_intent.create_intent_checker_llm(op_model_name)

            # Instantiate Event Registeration LLM + Tool Binding.
            self.agent_llm.create_event_registeration_llm(op_model_name, agent_tools)

            # Build State Graph
            self.agent_graph.build(self.agent_llm, self.agent_tools)

            # Initialise User Registeration Sub-Agent
            userreg_instr = self.instr_config["user_registeration"]
            await self.userreg_agent.initialize(userreg_instr)

            # Instantiat Registeration Lookup Sub-Agent
            reglkp_instr = self.instr_config["registeration_lookup"]
            reglkp_resp_instr = self.instr_config["registeration_lookup_response"]
            lookup_toolcall_instr = self.instr_config["lookup_toolcall_response"]
            await self.reglkp_agent.initialize(reglkp_instr, reglkp_resp_instr, lookup_toolcall_instr)

            # Launch Chat Agent
            resp_message = await self.agent_tools.launch(self.agent_graph, self.runevent_agent_instr)
            resp_message = (resp_message["out_messages"][-1]).content
            print("AI Response-1: ", resp_message)

            return(True)
        
        except Exception as exp:
            error = f"Error inside ChatAgent.initialize - {exp}"
            print(error, flush=True)
            raise Exception(error)
        
    # Chat Agent Run Entrypoint Function.
    # Prompt User for Input & Get Intent
    @traceable
    async def run(self, user_request: str) -> str:
        try:

            is_prompt_cached = False    # Init Prompt Caching to False

            # Check Intent of User Inputs
            user_intent = self.user_intent.check_user_intent(user_request, self.userintent_instr)
            print("User Intent:", user_intent, flush=True)

            # Check if Prompt Cached for this Intent & Get the AI Response
            if (user_intent in ["Register_User", "Check_Fees", "Check_Schedule"]):
                is_prompt_cached, cached_ai_response = self.user_intent.check_prompt_cache(user_intent)

            # If Cache Miss - Call OpenAI LLM with User Prompt
            # If Cache Hit  - Do Not Call LLM and Update State AIMessage
            if (is_prompt_cached == False):
                user_prompt = [HumanMessage(content=user_request)]
                resp_message = await self.agent_graph.ainvoke(user_prompt)
                resp_message = (resp_message["out_messages"][-1]).content

                # Update FAISS & Cache with User Prompt
                if (user_intent in ["Register_User", "Check_Fees", "Check_Schedule"]):
                    self.user_intent.cache_user_prompt(user_intent, resp_message)

                print("AI Response-1: ", resp_message)
                return(resp_message)

            elif (is_prompt_cached == True):
                message_list = [
                    HumanMessage(content=user_request),
                    AIMessage(content=cached_ai_response)
                ]
                self.agent_graph.update_state(message_list)

                print("AI Response-2: ", cached_ai_response)
                return(cached_ai_response)
            
            else:
                raise Exception("Error setting Prompt Caching Flag.")

        except Exception as exp:
            error = f"Error inside ChatAgent.run - {exp}"
            print(error, flush=True)
            raise Exception(error)
    

# Start the main program
# if __name__ == "__main__":
#    try:
#        chat_agent = ChatAgent()
#        asyncio.run(chat_agent.main())
#    except KeyboardInterrupt:
#        print("\nShutdown signal received. Cleaning up resources...", file=sys.stderr)
#        print("Server successfully stopped. Goodbye!", file=sys.stderr)
#    except Exception as exp:
#        print(f"Shutting down Running Event Agent due to Exception - {exp}", file=sys.stderr)

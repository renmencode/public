# Banking Customer Support AI Agent using Multi-Agent Architecture.<br>
# 
# Problem scenario:<br>
# 
# Modern digital banking platforms handle a high volume of customer service interactions, often through fragmented systems that struggle to personalize responses or provide timely status updates. The need for scalable, intelligent support systems has led to the emergence of AI-driven agents capable of parsing user sentiment, managing service records, and handling real-time queries. <br>
#  
# This project explores the development of a multi-agent GenAI system tailored for banking customer support workflows. The goal is to reduce manual effort, enhance customer satisfaction, and ensure timely response to support-related feedback and queries.
# 
# Project objective:<br>
# 
# This project aims to build: 
# - Classification of incoming user messages into feedback (positive or 
# negative) or queries 
# - Personalized responses based on classification and user sentiment 
# - Ticket tracking and updates through integration with a support database.
# 

# Solution Steps: <br>
# 1. Used two Langchain Agent
# 2. First Agent decides if the Fedback in Positive, Negative or a Query to check Ticket Status
# 3. If the feedback is Positive, appropriate Thank you Response is formatted and send back.
# 4. If the feedback is Negative, Agent asks if a ticket needs to be created.
# 5. If a new ticket needs to be created then, it goes ahead and creats a Ticket.
# 6. Primary key is User Name that is captured at the begining when Custmer Suppoer Agent come online  
# 7. If the User Input is a Query to List Tickets or Get Status of Ticket, then request is passed to a seconf Agent that make ToolCall to either List Ticket Tool or Get Status Tool.
# 8. All response of interactions are shown in the Test Case below
# 9. Models use for Interactions are - "llama3.1:8b-instruct-q4_K_M"
# 
# Test Cases: <br>
# User Input:  I am happy to get my firs Credit Card - Thank you <br>
# Assistant:  Thank you for the PositiveFeedback
# 
# User Input:  Can i get all tickets in my name ? <br>
# Inside List User Tickets <br>
# Assistant:  Number of Tickets for User: Menon - [471718, 299247]
# 
# User Input:  What is status of ticket number 299247 ? <br>
# Inside Ticket Status Lookup <br>
# Assistant:  Status of your ticket #299247 is currently marked as: CLOSED.
# 
# User Input:  Wha about 471718 ? <br>
# Inside Ticket Status Lookup <br>
# Assistant:  Status of your ticket #471718 is currently marked as: OPEN.
# 
# User Input:  I was not happy with he Teler Interaction when i visited back last week. I need to open a case. <br>
# Assistant:  Sorry to hear the Feedback, Should i create a Support Ticket to resolve your concern ? (Y/N) <br>
# Assistant:  We apologize for the inconvenience. A new ticket #128704 has been generated, and our team will follow up shortly. <br>
# 


# Import System Packages
import os, sys
import keyboard
import asyncio, json
from dotenv import load_dotenv
from pydantic import Field, create_model


# Import Langchain Agents Modules.
from langchain.agents import create_agent
from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage


# Import Langchain Ollama Chatbot Interfaces
from langchain_ollama import ChatOllama

# Import MCPClient
from fastmcp import Client

# Huggingface Cache Folder
hf_cache_folder = os.path.expanduser("~/.cache/huggingface/hub")

# Load Env Info
load_dotenv()


# Instantiate User Context Object

class UserContext:

    def __init__(self):
        self.user_name = ""

    # Get User Name
    def get_user_name(self):
        return(self.user_name)
    
    def set_user_name(self, user_name: str):
        self.user_name = user_name



# MCPClient Setup
# Instantiate MCPClient
class FastMCPClient:

    def __init__(self):
        self.mcp_client = Client("http://localhost:5000/mcp")

    # Connect to MCP Server
    async def connect(self):
        await self.mcp_client.__aenter__()

    # DisConnect to MCP Server
    async def disconnect(self):
        await self.mcp_client.__aexit__(None, None, None)

    # Ping MCP Server
    async def ping_mcp_server(self):
        print("Inside Ping MCP...", await self.mcp_client.ping())

    # Get Tool List
    async def list_mcp_tools(self):
        tools = await self.mcp_client.list_tools()
        return(tools)
    
    def get_mcp_client_conn(self):
        return(self.mcp_client)
    

# Instantiate FastMCPInvoker
class FastMCPInvoker:

    def __init__(self, tool_name: str, mcp_client_conn: object, user_context: UserContext):
        self.tool_name = tool_name
        self.mcp_client_conn = mcp_client_conn
        self.user_ctx = user_context                # Object is Only Populatd. 'user_name' is set later in the flow. 

    # Dynamic Invoke Function
    async def invoke(self, **kwargs) -> object:
        print("Inside FastMCP Invoker.....", self.tool_name)

        user_name = self.user_ctx.get_user_name()   # Lookup 'user_name' from the ContextObj when Invoked.         

        response = await self.mcp_client_conn.call_tool(
            self.tool_name,
            kwargs,
            meta = {
                "user_name": user_name
            }
        )

        if ("tool_resp" in response.content[0].text):
            resp_dict = json.loads(response.content[0].text)
            response = (
                "",
                resp_dict
            )
        else:
            resp_dict = json.loads(response.content[0].text)
            raise RuntimeError(resp_dict)

        return(response)



# Instantiate FastMCP <-> Langchain Adapter
# Purpose to convert FastMCP Tool Lookup Response to LangChain Strucured Tool Format

class LangchainAdapter:

    def __init__(self, mcp_client: FastMCPClient):
        self.fastmcp_client = mcp_client

    # Convert to Langchain Schema
    def create_args_schema(self, tool_inp_schema: dict) -> tuple[str, object]:

        fields = {}

        type_map = {
            "string": str,
            "integer": int
        }

        properties = tool_inp_schema["properties"]
        args_title = tool_inp_schema["title"]

        for field_name, field_info in properties.items():
            field_type = field_info['type']

            py_data_type = type_map[field_type]
            field_obj = Field(description=field_info["title"])

            fields[field_name] = (
                py_data_type,
                field_obj
            )

        return(args_title, create_model(args_title, **fields))


    # Generate Langchain Tool List for Registering with Agent.
    def generate_langchain_tool_spec(self, mcp_tool_list: list, user_context: UserContext) -> list[StructuredTool]:

        tools = []

        mcp_client_conn = self.fastmcp_client.get_mcp_client_conn()

        for mcp_tool in mcp_tool_list:
            mcp_tool_inp_schema = mcp_tool.inputSchema
            args_schema = self.create_args_schema(mcp_tool_inp_schema)

            # print("Tool Schema: ", mcp_tool_inp_schema)
            # print("Arguments Schema: ", args_schema)

            # Create Instance of MCPInvoker
            mcp_invoker = FastMCPInvoker(
                mcp_tool.name,
                mcp_client_conn,
                user_context
            )

            # Build the Langchain StructureTool
            # Add the MCPInvoker Instance Invoke Func 
            struct_tool = StructuredTool.from_function(
                name = mcp_tool.name,
                description = mcp_tool.description,
                args_schema = args_schema[1],
                return_direct = True,
                response_format = "content_and_artifact",
                coroutine = mcp_invoker.invoke
            )

            tools.append(struct_tool)

        return(tools)
               

# Building a Feedback Handler & Query Tools
# Positive & Negative Feedback Handler Services.

class FeedbackHandler:

    def __init__(self):
        pass

    # Positive Feedback Handler 
    async def positive_feedback_handler(self, user_name: str) -> str:

        response = f"Thank you for your kind words, {user_name}! We're delighted to assist you"
        return(response)

    # Negative Feedback Handler
    async def negative_feedback_handler(self, fastmcp_client: FastMCPClient, user_context: UserContext) -> str:

        try:
            mcp_client_conn = fastmcp_client.get_mcp_client_conn()
            mcp_invoker = FastMCPInvoker("create_user_ticket", mcp_client_conn, user_context)
            response = await mcp_invoker.invoke()                 # Invoke MCP Worker with No Args

            return(response[1]["tool_resp"])
        
        except Exception as exp:
            error_resp = f"Error Creating Negative Feedback - {exp}"
            print(error_resp, flush=True)
            return(error_resp)


# AgentLLM Class to Instantiate LLMs

class AgentLLM:

    def __init__(self):
        pass

    # Instantiate Customer Support LLM
    def create_custsupport_llm(self, model_name: str) -> ChatOllama:

        ol_custsupport_llm = ChatOllama(
            model = model_name,
            temperature = 0.0,
            num_predict = 512,
            num_ctx = 2048,
            model_kwargs = {
                "repeat_penalty": 1.1,
                "options": {
                    "use_cache": False,
                    "use_mmap": False
                }
            }   
        )

        return(ol_custsupport_llm)

    # Instantiate User Query LLM
    def create_userquery_llm(self, model_name: str, tool_list: list[StructuredTool]) -> ChatOllama:

        ol_userquery_llm = ChatOllama(
            model = model_name,
            temperature = 0.0,
            num_predict = 512,
            num_ctx = 2048,
            model_kwargs = {
                "repeat_penalty": 1.1,
                "options": {
                    "use_cache": False,
                    "use_mmap": False
                }
            }   
        )

        ol_userquery_llm_with_tools = ol_userquery_llm.bind_tools(tool_list)

        return(ol_userquery_llm_with_tools)


# AgentFactory helps to create Agent for Customer Support and Query

class AgentFactory:

    def __init__(self):
        self.agent_llm = AgentLLM()

    # Agent decide which tool to call - User Ticket Status or User Ticket List
    def initialise_userquery_agent(self, model_name: str, tool_list: list[StructuredTool]) -> object:

        userquery_llm = self.agent_llm.create_userquery_llm(model_name, tool_list)

        agent_instruction = """
        Role: You are a User Query Answering Assistant.

        Goals:
        1. Analyze the User Query and decide the right Tool to Call.
    
        Available Tools:
        1. list_user_tickets: Tool used to get/retrieve List of Tickets for a User.
        2. lookup_ticket_status: Tool used to get/lookup Status of a User's Ticket.

        Examples:
            User Input: Could you check the status of ticket 650932 ?
            Assistant: Make ToolCall to 'lookup_ticket_status' Tool.

            User Input: Get the list of my tickets ?
            Assistant: Make ToolCall to 'list_user_tickets' Tool.
        """

        agent = create_agent(
            model = userquery_llm,
            tools = tool_list,
            system_prompt=SystemMessage(content=agent_instruction)
            )
        
        return(agent)


    # Creating an Customer Support Inferencing Agent
    # Agent understand the Sentiment of User Input. Positive, Negative, Query
    def initialize_custsupport_agent(self, model_name: str) -> object:

        custsupport_llm = self.agent_llm.create_custsupport_llm(model_name)

        agent_instruction = """
        Role: You are a Banking Customer Support Assistant.

        Goals:
        1. Analyze the Semantic Meaning of the User Request and CLASSIFY it into EXACTLY ONE of the Allowed TASK.
        2. If the User's intent Matches with Any One Task from the Task List, return that Value.
        3. If the User's intent Does Not Match with value in Task List, respond with UNKNOWN_TASK.
    
        Allowed Tasks:
        1. QueryTask: ONLY IF User Request is to Get Ticket Status or Get List of Tickets.
        2. PositiveFeedback: ONLY IF User is expressing appreciation, satisfaction, or positive sentiment.
        3. NegativeFeedback: ONLY IF User is expressing dissatisfaction, frustration, or a complaint.

        Examples:
            User Input: Thanks for sorting out my net banking login issue.
            Assistant: PositiveFeedback

            User Input: My debit card replacement still hasn't arrived.
            Assistant: NegativeFeedback

            User Input: Could you check the status of ticket 650932 ?
            Assistant: QueryTask

            User Input: What is the Weather today ?
            Assistant: UNKNOWN_TASK
        """

        agent = create_agent(
            model = custsupport_llm,
            system_prompt=SystemMessage(content=agent_instruction)
            )
        
        return(agent)


    # Format Agent Response
    def format_agent_response(self, ai_msg):
        print(ai_msg['messages'][1])
        ai_msg = (ai_msg['messages'][1]).content.split("<|im_start|>assistant")[-1]
        return(ai_msg.strip())


# Initiate Chat with Agent

class ChatAgent:

    def __init__(self):
        self.exit_flag = False


    # Press 'Escape' Key to End the While Loop below
    def on_key_press(event):
        if event.name == 'esc':
            print("Escape key pressed! Exiting input.")
            global exit_flag
            exit_flag = True
            return True             # Stop the keyboard listener

    # Chat Agent Initialization
    async def initialize_chat_agent(self):
        
        print("Initializing Chat Agent...", flush=True)

        # Create Empty User Context
        self.user_context = UserContext()

        # Initialize FastMCPClient & Get ToolList
        self.fastmcp_client = FastMCPClient()
        await self.fastmcp_client.connect()
        mcp_tool_list = await self.fastmcp_client.list_mcp_tools()

        # Initialise FeedbackHandler
        self.feedback_handler = FeedbackHandler()

        # Create Langchain Adapter. Get Info on exposed MCP Tools.
        self.langchain_adapter = LangchainAdapter(self.fastmcp_client)
        tool_list = self.langchain_adapter.generate_langchain_tool_spec(mcp_tool_list, self.user_context)

        llama_model = "llama3.1:8b-instruct-q4_K_M"
        self.agent_factory = AgentFactory()
        self.custsupport_agent = self.agent_factory.initialize_custsupport_agent(llama_model)
        self.userquery_agent = self.agent_factory.initialise_userquery_agent(llama_model, tool_list)


    # Chat Agent Run Entrypoint Function.
    async def run(self):

        # Initiate User Conversation & Greetings
        user_input = input("Hi, I am a Bank Customer Feedback Agent, Please Enter your First Name: ")
        user_name = user_input
        self.user_context.set_user_name(user_name)

        # User Interaction Loop.
        greeting_flag = True

        while(not self.exit_flag):
            if (greeting_flag == True):
                print(f"Hello {user_name}")
                user_input = input(f"Hello {user_name}, Can I help with Question related to Tickets or Feedback if any: ")
                greeting_flag = False
            else:
                user_input = input("Can an i help with anymore Question realted to Tickets or Feedback: ")

            print("User Input: ", user_input, flush=True)
            if (not self.exit_flag):
                user_prompt = {"messages": [HumanMessage(content=user_input)]}
                agent_response = self.custsupport_agent.invoke(user_prompt)
                agent_task = self.agent_factory.format_agent_response(agent_response)
                if ('PositiveFeedback' in agent_task):                                      # PositiveFeedback Processing
                    print("Assistant: ", await self.feedback_handler.positive_feedback_handler(user_name), flush=True)
                elif ('NegativeFeedback' in agent_task):                                    # NegativeFeedback Processing
                    print("Assistant: ", "Sorry to hear the Feedback, Should i create a Support Ticket to resolve your concern ? (Y/N)", flush=True)
                    user_confirm = input(f"{user_name} - Create a Support Ticket ? (Y/N)")
                    if ((user_confirm.upper() == "Y") or (user_confirm.upper() == "YES")):
                        print("Assistant: ", await self.feedback_handler.negative_feedback_handler(self.fastmcp_client, self.user_context), flush=True)
                elif ('QueryTask' in agent_task):                                           # User Query Processing
                    userquery_agent_resp = await self.userquery_agent.ainvoke(user_prompt)  # .ainvoke(..) for Async Invoke
                    userquery_agent_tool_resp = userquery_agent_resp['messages'][-1]
                    if isinstance(userquery_agent_tool_resp, ToolMessage):
                        print("Assistant: ", userquery_agent_tool_resp.artifact.get("tool_resp"), flush=True)
                else:
                    print("Assistant: ", "Not able to understand your Request. I can hlp you with Support Ticket Information or Feedbacks.", flush=True)    # Unknown Task Processing
            else:
                await self.fastmcp_client.disconnect()

    async def main(self):
        await self.initialize_chat_agent()
        await self.run()

# Start the main program
if __name__ == "__main__":
    try:

        chat_agent = ChatAgent()
        # keyboard.on_press(chat_agent.on_key_press)
        asyncio.run(chat_agent.main())

    except KeyboardInterrupt:
        print("\nShutdown signal received. Cleaning up resources...", file=sys.stderr)
        print("Server successfully stopped. Goodbye!", file=sys.stderr)


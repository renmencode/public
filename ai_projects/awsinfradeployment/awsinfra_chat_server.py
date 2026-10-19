# Class Late Binding Annotaion
from __future__ import annotations 

# Import Python Native Lib
import os, sys
import json
import asyncio
import numpy as np
import uuid as uuid
from dotenv import load_dotenv

# Import Uvicorn Server
# Import FastAPI Framework
import uvicorn  
from fastapi import FastAPI

# Pydantic BaseModel class
from pydantic import BaseModel, Field

# Import LangSmith for Tracing
from langsmith import traceable

# Custom Classes
from awsinfra_deployment_agent import AWSInfraAgent

# Load Env Variables
load_dotenv()

# Retrieve Global Values
chat_uri = os.getenv("awsinfra_chat_server.post.chat_uri")

# Init Langsmith Tracing
#os.environ["LANGSMITH_TRACING"] = "true"
#os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
#os.environ["LANGSMITH_PROJECT"] = os.getenv("runevent_monitor_app")


# Chat Request Class
class ChatRequest(BaseModel):
    interrupt_flag: bool | None = None
    interrupt_name: str | None = None
    interrupt_payload: dict | None = None
    message: str | None = None

# Chat Response Class
class ChatResponse(BaseModel):
    interrupt_flag: bool | None = None
    interrupt_name: str | None = None
    message: str

# ChatBot Server Implementation
class ChatBotServer:

    def __init__(self):
        self.deployment_agent = AWSInfraAgent()
        self.server_host = os.getenv("awsinfra_chat_server.host")
        self.server_port = int(os.getenv("awsinfra_chat_server.port"))

    # Implement 'chat' interface
    #@traceable(name="runevent_chat")
    async def chat(self, chat_req: ChatRequest) -> ChatResponse:
        try:
            print("User Request - ", chat_req)

            if (chat_req.interrupt_flag == True):
                payload = {
                    "decisions": [
                        {
                            "type": "edit",
                            "edited_action": {
                                "name": chat_req.interrupt_name,
                                "args": chat_req.interrupt_payload
                            }
                        }
                    ]
                }
                chat_resp = await self.deployment_agent.run(payload)

            else:
                chat_resp = await self.deployment_agent.run(chat_req.message)

            return(chat_resp)
        
        except Exception as exp:
            error = f"Error inside ChatBotServer.chat - {exp}"
            print(error, flush=True)
            raise Exception(error)

    # Star the ChatBot Server
    async def main(self) -> bool:
        try:
            uvi_config = None
            uvi_server = None

            is_agent_initialized = await self.deployment_agent.initialize()

            # Start ChatBot Server
            # UVI Server Runs Synchronously by default.
            if (is_agent_initialized == True):
                uvi_config = uvicorn.Config(
                    app,
                    host=self.server_host,
                    port=self.server_port
                )
                uvi_server = uvicorn.Server(uvi_config)
                await uvi_server.serve()
                return(True)
            else:
                raise Exception("Chat Agent Initiaization Failed.")

        except Exception as exp:
            error = f"Error inside ChatBotServer.main - {exp}"
            print(error, flush=True)
            raise Exception(error)


# Instantiate FastAPI
# Register ChatBotServer & Context Path
app = FastAPI()
chatbot = ChatBotServer()
app.post(chat_uri)(chatbot.chat)

# Entrypoint for ChatBot Server 
if __name__ == "__main__":
    try:
        asyncio.run(chatbot.main())
    except KeyboardInterrupt:
        print("\nShutdown signal received. Cleaning up resources...", file=sys.stderr)
        print("AWSInfra Deployment Agent Chat Server successfully stopped. Goodbye!", file=sys.stderr)
    except Exception as exp:
        print(f"Shutting down AWSInfra Deployment Agent due to Exception - {exp}", file=sys.stderr)

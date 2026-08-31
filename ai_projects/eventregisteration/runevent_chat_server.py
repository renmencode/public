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
from pydantic import BaseModel

# Import LangSmith for Tracing
from langsmith import traceable

# Custom Classes
from runevent_registration_agent import ChatAgent

# Load Env Variables
load_dotenv()

# Retrieve Global Values
chat_uri = os.getenv("runevent_chat_server.post.chat_uri")

# Init Langsmith Tracing
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("runevent_monitor_app")


# Chat Request Class
class ChatRequest(BaseModel):
    message: str

# Chat Response Class
class ChatResponse(BaseModel):
    message: str

# ChatBot Server Implementation
class ChatBotServer:

    def __init__(self):
        self.is_initialized = None
        self.chat_agent = ChatAgent()
        self.server_host = os.getenv("runevent_chat_server.host")
        self.server_port = int(os.getenv("runevent_chat_server.port"))

    # Implement 'chat' interface
    @traceable(name="runevent_chat")
    async def chat(self, chat_req: ChatRequest) -> ChatResponse:
        print("User Request - ", chat_req.message)

        chat_resp = await self.chat_agent.run(chat_req.message)

        return({
            "message": chat_resp
        })

    # Implement 'Init' interface
    @traceable(name="runevent_init")
    async def initialize(self):
        self.is_initialized = await self.chat_agent.initialize()      # Init Chat Agent


    # Star the ChatBot Server
    async def main(self) -> bool:
        try:
            uvi_config = None
            uvi_server = None

            await self.initialize()

            # Start ChatBot Server
            # UVI Server Runs Synchronously by default.
            if (self.is_initialized == True):
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
        print("Server successfully stopped. Goodbye!", file=sys.stderr)
    except Exception as exp:
        print(f"Shutting down Running Event Agent due to Exception - {exp}", file=sys.stderr)

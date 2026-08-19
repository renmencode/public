# Class Late Binding Annotaion
from __future__ import annotations 

import os, sys
import asyncio
from dotenv import load_dotenv

# Import Strealmit
import streamlit as st

# Import HTTP Request Module
import requests

# Load Env Variables
load_dotenv()

# Chat Client Class
class RunEventChatClient:

    def __init__(self):
        st.title("User Registeration Chat UI.")

    # Post Message
    def post_message(self, prompt: str):
        try:
            payload = {
                "message": prompt
            }

            response = requests.post(
                os.getenv("runevent_chat_server_url"),
                json=payload
            )
            chat_resp = response.json()

            return(chat_resp["message"])

        except Exception as exp:
            error = f"Error inside RunEventChatClient.post_message - {exp}"
            print(error, flush=True)
            raise Exception(error)

    # Chat with Message
    def chat(self):
        try:
            if "messages" not in st.session_state:          # Check if Session has 'messages'
                st.session_state.messages = []
    
            prompt = st.chat_input("Your Message")
            if (prompt is not None):
                chat_resp = self.post_message(prompt)
                st.session_state.messages.append(chat_resp)
                st.write(st.session_state.messages)

        except Exception as exp:
            error = f"Error inside RunEventChatClient.post_message - {exp}"
            print(error, flush=True)
            raise Exception(error)

if __name__ == "__main__":
    try:
        runevent_chat_client = RunEventChatClient()
        runevent_chat_client.chat()

    except KeyboardInterrupt:
        print("\nShutdown signal received. Cleaning up resources...", file=sys.stderr)
        print("Server successfully stopped. Goodbye!", file=sys.stderr)
    except Exception as exp:
        print(f"Shutting down Running Event Agent due to Exception - {exp}", file=sys.stderr)

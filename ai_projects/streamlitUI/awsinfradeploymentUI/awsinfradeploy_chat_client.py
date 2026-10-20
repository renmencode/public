# Class Late Binding Annotaion
from __future__ import annotations 

import os, sys
import asyncio
from dotenv import load_dotenv

# Import Strealmit
import re
import streamlit as st

# Import HTTP Request Module
import requests

# Load Env Variables
load_dotenv()


# Chat Client Class
class AWSInfraDeployChatClient:

    def __init__(self):
        st.title("AWS Infra Deployment Chat UI.")

        self.chat_server_url = os.getenv("awsinfradeploy_chat_server_url")      # URL from ENV.

    # Post Message
    def post_message(self, payload: dict):
        try:

            response = requests.post(
                url=self.chat_server_url,
                json=payload
            )
            chat_resp = response.json()

            print("\nCHAT_RESP:", chat_resp, flush=True)

            # Set Streamlit HITL Interrupt Flag
            if (chat_resp.get("interrupt_flag", False)):
                st.session_state.hitl_flag = True

            return(chat_resp)

        except Exception as exp:
            error = f"Error inside AWSInfraDeployChatClient.post_message - {exp}"
            print(error, flush=True)
            raise Exception(error)

    # User Interrupt Handler
    def handle_interrupt(self, interObj: dict) -> dict:
        try:
            print("\nInside HANDLE INTERRUPT", interObj, flush=True)
            # Deploy Template HITL Interrupt
            if (interObj["interrupt_name"] == "deploy_template"):
                stack_name = st.text_input("Stack Name")
                if (stack_name is not None):
                    interrupt_args = {
                        "stack_name": stack_name
                    }

                if (st.button(label="Submit")):
                    interrupt_payload = {
                        "user_input": interrupt_args
                    }

                    return(interrupt_payload)
                
                else:
                    return(None)
                
            else:
                return(None)
            
        except Exception as exp:
            error = f"Error inside AWSInfraDeployChatClient.hendle_interrupt - {exp}"
            print(error, flush=True)
            raise Exception(error)

    # Initialize Chat with Selutation
    def initialize(self):
        try:
            # Init State Variables 
            st.session_state.messages = []
            st.session_state.hitl_data = None
            st.session_state.hitl_flag = False

            prompt_dict = {
                "message": "Hello"
            }

            chat_resp = self.post_message(prompt_dict)

            st.session_state.messages.append(chat_resp["message"])
            st.write(st.session_state.messages)

        except Exception as exp:
            error = f"Error inside AWSInfraDeployChatClient.initialize - {exp}"
            print(error, flush=True)
            raise Exception(error)


    # Chat with Message
    def chat(self):
        try:
            if "messages" not in st.session_state:          # Check if Session has 'messages'
                self.initialize()

            # HITL User Interaction.
            # When forced to ReRun - Contols comes here.
            if (st.session_state.hitl_flag == True):
                st.write(st.session_state.messages)      # User Input Appears below Rolling Messages

                hitl_dict = self.handle_interrupt(st.session_state.hitl_data)
                if (hitl_dict is not None):
                    hitl_args = hitl_dict["user_input"]
                    hitl_desplay = "User Input: " + hitl_args["stack_name"]
                    st.session_state.messages.append(hitl_desplay)
                    st.session_state.hitl_flag = False

                    chat_resp = self.post_message({
                        "interrupt_flag": True,
                        "interrupt_name": st.session_state.hitl_data["interrupt_name"],
                        "interrupt_payload": hitl_dict
                    })
                    # Append ChatResponse to Session.
                    st.session_state.messages.append(chat_resp["message"])
                    st.write(st.session_state.messages)             # Write the Session to UI
                    
                else:
                    print("\nHITL User Input is None. Waiting for User Input")

            else:
                pass

            # Regular User Interations
            prompt = st.chat_input(placeholder="Your Message")
            if (prompt is not None):
                st.session_state.messages.append(prompt)        # Append User Prompt to Session.
                prompt_dict = {
                    "message": prompt
                }

                chat_resp = self.post_message(prompt_dict)      # Post the Message to Server

                if (st.session_state.hitl_flag == True):        # If hitl_flag is True, Force Streamlit Re-Run
                    st.session_state.hitl_data = chat_resp
                    st.session_state.messages.append(chat_resp["message"])                      
                    st.rerun()                                  # No Write is done, happens at the begining of Re-Run
                else:
                    st.session_state.messages.append(chat_resp["message"])     # Append ChatResponse to Session.
                    st.write(st.session_state.messages)             # Write the Session to UI

        except Exception as exp:
            error = f"Error inside AWSInfraDeployChatClient.chat - {exp}"
            print(error, flush=True)
            raise Exception(error)

if __name__ == "__main__":
    try:
        awsinfradeploy_chat_client = AWSInfraDeployChatClient()
        awsinfradeploy_chat_client.chat()

    except KeyboardInterrupt:
        print("\nShutdown signal received. Cleaning up resources...", file=sys.stderr)
        print("AWSInfra Deployment UI Server successfully stopped. Goodbye!", file=sys.stderr)
    except Exception as exp:
        print(f"Shutting down AWSInfra Deployment UI due to Exception - {exp}", file=sys.stderr)

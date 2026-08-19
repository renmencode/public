# Class Late Binding Annotaion
from __future__ import annotations 

# Import Python Native Lib
import os, sys
import keyboard
import asyncio
import numpy as np
import uuid as uuid
from dotenv import load_dotenv

from typing_extensions import Literal, TypedDict

# Import Pandas
import pandas as pd

# Import FAISS Store
import faiss

# Import Sentence Ebedding Transformer
from sentence_transformers import SentenceTransformer

# Import LangGraph & LangChain packages
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, AnyMessage, RemoveMessage

# Import Langchain AI Models
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# Huggingface Local Cache folder
hf_cache_folder = os.path.expanduser("~/.cache/huggingface/hub")

# Load Env Variables
load_dotenv()


# Intent Message Dict
class IntentState(TypedDict):
    int_message: list[AnyMessage] 


# Class used to Check Intent of User's Request
class UserIntent:

    def __init__(self):
        self.op_intentchecker_llm = None
        self.prompt_cache_df = pd.DataFrame(columns=["Intent", "AIResponse"])

    # Create Intent Checking Ollama / Qwen LLM
    def create_intent_checker_llm(self, model_name: str) -> object:

        self.op_intentchecker_llm = ChatOpenAI(
            model=model_name, 
            temperature=0.0, 
            max_retries=2
        )

    # Check for User Intent
    def check_user_intent(self, user_query: str, system_instruction: str) -> Literal["Register_User", "Check_Fees", "Check_Schedule", "NO_INTENT"]:

        system_prompt_intent = system_instruction

        intent_llm = self.op_intentchecker_llm

        intent_array = []
        intent_array = [SystemMessage(content=system_prompt_intent),
                        HumanMessage(content=user_query)
        ]

        intent_prompt: IntentState = IntentState({"int_message": intent_array})
        ai_response = intent_llm.invoke(intent_prompt["int_message"])

        return(ai_response.content)

    # Update Usr Prompt to FAISS
    # Update Prompt Cache (Pandas Store) for Lookup.
    def cache_user_prompt(self, intent: str, ai_resp: str) -> bool:
        try:
            # Add Data to Data Frame
            self.prompt_cache_df.loc[len(self.prompt_cache_df)] = [intent, ai_resp]

            # Create Sentence Embedding. Convert String-to-String inside a List before embedding is done.
            # sentence_embedding = self.model.encode([prompt], batch_size=32, output_value="sentence_embedding", convert_to_numpy=True, device="cpu")
            # print("Embedding Shape: ", sentence_embedding.shape)

            # Append Sentence Embeddings
            # self.fs_vector_index.add(sentence_embedding)
            # print("Vector Count: ", self.fs_vector_index.ntotal)

            return(True)

        except Exception as exp:
            error = f"Error inside UserIntent.cache_user_prompt - {exp}"
            print(error, flush=True)
            raise Exception(error)
        
    # Check if User Prompt is Cached
    def check_prompt_cache(self, intent: str) -> tuple[str, str]:
        try:
            response = None
            is_cached = False

            if (not self.prompt_cache_df.empty):
                response = self.prompt_cache_df[self.prompt_cache_df['Intent'] == intent]["AIResponse"]
                if (response is not None) and (not response.empty):
                    is_cached = True
                    response = response.values[0]

            return(is_cached, response)

        except Exception as exp:
            error = f"Error inside UserIntent.check_prompt_cache - {exp}"
            print(error, flush=True)
            raise Exception(error)

# Class Late Binding Annotaion
from __future__ import annotations 

# Import Python Native Lib
import os, sys
import keyboard
import numpy as np
import uuid as uuid
from dotenv import load_dotenv

# Import Pandas
import pandas as pd

# Import Sentence Ebedding Transformer
from sentence_transformers import SentenceTransformer

# Import RAG DB FAISS
import faiss

# Huggingface Local Cache folder
hf_cache_folder = os.path.expanduser("~/.cache/huggingface/hub")

# Run Event FAQ File
faq_csv_file = "C:\\RanjithC\\AIProjects\\PromptEngg\\mcpservers\\data\\runevent\\runevent_faq.csv"

# Load Env Variables
load_dotenv()


# RunEvent FAQ Class.
class RunEventFAQ:

    def __init__(self):
        self.model = SentenceTransformer(
            'all-MiniLM-L6-v2',
            similarity_fn_name='cosine', 
            cache_folder = hf_cache_folder,
            local_files_only=True,
            device='cpu'
        )
        self.fs_vector_index = None
        self.pd_data = None

    
    # Load Event FAQ to FAISS Vector Store
    def load_faq_vectorstore(self) -> bool:

        try:
            # Create a DataFrame of Two Columns
            self.pd_data = pd.read_csv(faq_csv_file, sep="|")
            faq_content = self.pd_data['Questions'].tolist()

            #Create Sentence Embedding 
            sentence_embedding = self.model.encode(faq_content, batch_size=32, output_value="sentence_embedding", convert_to_numpy=True, device="cpu")
            print("Embedding Shape: ", sentence_embedding.shape)

            # Initialise FAISS Vector Store & Load
            self.fs_vector_index = faiss.IndexFlatL2(sentence_embedding.shape[1])
            self.fs_vector_index.add(sentence_embedding)
            print("Vector Count: ", self.fs_vector_index.ntotal)

            return(True)
       
        except Exception as exp:
            print(f"Loading of FAQ to VectorStore Failed - {exp}")
            return(False)


    # Check in Event FAQ Tool. Binding of this to LLM is done in Parent Class
    def search_faq(self, user_query: str) -> str:
        """Provides ability to Search the Frequently Asked Questions (FAQ) database to answer User's Query on the Running Event.
        
        Args: 
            user_query: The exact question or request from the user in plain text.

        Returns: 
            response: A short, factual answer to the question from the FAQ database. 

        Examples of User Query:
            1. I need to Register for the Running Event.
            2. I need to add a person for the Event.
            3. How do i get information to Sign-up for Running Event ?
            4. Can i get information to Register for the Event ?
            5. Get me the Procedure to Signup or Register for the Event.

        """
        
        try:
            query_embedding = self.model.encode(user_query, batch_size=32, output_value="sentence_embedding", convert_to_numpy=True, device="cpu")

            query_embedding = query_embedding[np.newaxis, :]                        # query_embedding to a 2D Vector of 1 Row and 1 column
            top_dist, top_pos = self.fs_vector_index.search(query_embedding, k=5)   # Independent Array of Eucl dist and Pos
            
            print("Top Euclidian Dist: ", top_dist)
            print("Top Answers: ", top_pos)

            top_idx = top_pos[0][0]                             # Top Result Index Number
            response = self.pd_data['Answers'][top_idx]

            return(response)
        
        except Exception as exp:
            print(f"Loading of FAQ to VectorStore Failed - {exp}")
            return(False)

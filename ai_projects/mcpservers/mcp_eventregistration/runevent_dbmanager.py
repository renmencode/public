# Class Late Binding Annotaion
from __future__ import annotations 

# Import Python Native Lib
import os, sys
import asyncio
import uuid as uuid
from dotenv import load_dotenv

# Import SQLLite
import sqlite3

load_dotenv()


# DB Manager Class
class DBManager:

    def __init__(self):
        self.sql_conn = None
        self.sqllite_db_file = os.getenv("runevent_sqllitedb_file")

    # Create DB manager Connection
    def create_sqllite_conn(self) -> sqlite3.Connection:
        try:
            self.sql_conn = sqlite3.connect(self.sqllite_db_file, check_same_thread=False)    # Tell SQLLite to run on multiple threads.
            
            cursor = self.sql_conn.cursor()
            cursor.execute(
                """
                    CREATE TABLE IF NOT EXISTS user_registeration (
                    reg_id INTEGER, first_name TEXT, last_name TEXT, age INTEGER, email TEXT, phone_number TEXT
                    )
                """
            )
            self.sql_conn.commit()
            cursor.close()

            return(True)
        
        except Exception as exp:
            error = f"Creation of SQL DB Failed - {exp}"
            print(error, flush=True)
            raise Exception(error)

    # Get DB Manager Connection
    def get_sqllite_conn(self):
        return(self.sql_conn)
# User Ticket Management Service

# Import System Packages
import os
import random
import keyboard
from dotenv import load_dotenv

# Import SQLLite
import sqlite3

# Import FastMCP Context Class
from mcp.server.fastmcp import Context

# Connect to SQLLite DB Path
sqllite_db_file = "C:\\RanjithC\\AIProjects\\PromptEngg\\mcpservers\\data\\custsupport\\customer_support.db"


class DBManager:

    def get_sqllite_conn(self) -> sqlite3.Connection:
        try:
            sql_conn = sqlite3.connect(sqllite_db_file, check_same_thread=False)    # Tell SQLLite to run on multiple threads.
            
            cursor = sql_conn.cursor()
            cursor.execute(
                """
                    CREATE TABLE IF NOT EXISTS support_tickets (
                    user_id TEXT, ticket_no INTEGER, status TEXT
                    )
                """
            )
            sql_conn.commit()
            cursor.close()

            return(sql_conn)
        
        except Exception as exp:
            print(f"Creation of SQL DB Failed - {exp}", flush=True)
            return(None)


class TicketManager:

    out_struct = {}

    def __init__(self):
        self.dbManager = DBManager()

    # Create a New Ticket for the User
    # MCP Server always return Str / Int / Dict type 
    def create_user_ticket(self, mcp_ctx: Context) -> dict:
        """ This Tool is used to Create Tickets for a User. 

        Return: Ticket Create Acknowledgement.
        """

        cursor = None
        print("Inside List User Tickets", flush=True)

        user_name = mcp_ctx.request_context.meta.user_name       # 'user_name' from Context
        print("User Name: ", user_name)

        try:
            ticket_no = random.randint(100000, 999999)

            sql_conn = self.dbManager.get_sqllite_conn()
            cursor = sql_conn.cursor()
            cursor.execute("INSERT INTO support_tickets (user_id, ticket_no, status) VALUES (?, ?, ?)",
                        (user_name, ticket_no, "open")
                        )
            sql_conn.commit()

            response = f"We apologize for the inconvenience. A new ticket #{ticket_no} has been generated, and our team will follow up shortly."
            
            return({"tool_resp": response})
        
        except Exception as exp:
            exp_resp = f"Error Creating User Ticket - {exp}"
            print(exp_resp, flush=True)
            return({"error_resp": exp_resp})
        finally:
            if (cursor is not None):
                cursor.close()

    # Get List of Tickets associated with User.
    # MCP Server always return Str / Int / Dict type 
    def list_user_tickets(self, mcp_ctx: Context) -> dict:
        """ This Tool is used to Get List of Tickets for the User. 

        Return: List of Tickets for the User.
        """
        
        cursor = None
        print("Inside List User Tickets", flush=True)

        user_name = mcp_ctx.request_context.meta.user_name

        try:
            sql_conn = self.dbManager.get_sqllite_conn()
            cursor = sql_conn.cursor()
            cursor.execute("SELECT ticket_no FROM support_tickets WHERE user_id = ?", 
                        (user_name,)
                        )

            record = cursor.fetchall()
            if (record == None):
                response = f"No Records Found for User: {user_name}."
            else:
                record_list = [row[0] for row in record]
                response = f"Number of Tickets for User: {user_name} - {record_list}"

            return({"tool_resp": response})

        except Exception as exp:
            exp_resp = f"Error listing User Tickets - {exp}"
            print(exp_resp, flush=True)
            return({"error_resp": exp_resp})
        finally:
            if (cursor is not None):
                cursor.close()

    # Get Status of a User's Ticket
    # MCP Server always return Str / Int / Dict type 
    def lookup_ticket_status(self, ticket_no: int, mcp_ctx: Context) -> dict:
        """ This Tool is used to Get the Status of User's Ticket. 

        Args: 
            ticket_no: Ticket Number for checking its Status.
 
        Return: Status of User's Ticket.
        """

        cursor = None
        print("Inside Ticket Status Lookup", flush=True)

        user_name = mcp_ctx.request_context.meta.user_name

        try:
            sql_conn = self.dbManager.get_sqllite_conn()
            cursor = sql_conn.cursor()
            cursor.execute("SELECT status FROM support_tickets WHERE user_id = ? AND ticket_no = ?", 
                        (user_name, ticket_no)
                        )
        
            record = cursor.fetchone()
            if (record == None):
                response = f"No Records Found for User: {user_name} and TicketNo: {ticket_no}"
            else:
                response = f"Status of your ticket #{ticket_no} is currently marked as: {record[0].upper()}."

            return({"tool_resp": response})
        
        except Exception as exp:
            exp_resp = f"Error Looking-up Ticket Status - {exp}"
            print(exp_resp, flush=True)
            return({"error_resp": exp_resp})
        finally:
            if (cursor is not None):
                cursor.close()

    
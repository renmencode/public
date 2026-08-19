# FastMCP Server for User Ticket Management

# Import System Packages 
import os, sys
from dotenv import load_dotenv

# Import FastMCP
from mcp.server.fastmcp import FastMCP

# Import External Modules
from runevent_faq_service import RunEventFAQ
from runevent_notify_service import RunEventNotifyUser
from runevent_usrreg_service import RunEventUserRegistration

load_dotenv()


# Create MCP Server
class MCPServer:

    def __init__(self, server_host: str, server_port: int):
        print("MCP Initialization...")
        server_host = os.getenv("mcp_runevent.host")
        server_port = os.getenv("mcp_runevent.port")

        self.mcp = FastMCP(
            name = "RunEventTools",
            host = server_host,
            port = server_port,
            stateless_http = True
        )                                                   # Initialise FastMCP
        
        self.runevent_faq = RunEventFAQ()                       # Initialize FAQ Tool
        self.runevent_notify = RunEventNotifyUser()
        self.runevent_userreg = RunEventUserRegistration()      # Initialize USerReg Tool

    # Register Tool in MCP Server
    def registerTools(self):
        print("Tool Registration...")
        self.mcp.tool(name="search_faq")(self.runevent_faq.search_faq)
        self.mcp.tool(name="validate_user")(self.runevent_userreg.validate_user)
        self.mcp.tool(name="register_user")(self.runevent_userreg.register_user)
        self.mcp.tool(name="notify_user")(self.runevent_notify.notify_user)
        self.mcp.tool(name="list_users")(self.runevent_userreg.list_registered_users)
        self.mcp.tool(name="lookup_user")(self.runevent_userreg.lookup_registered_user)

    # Trigger MCP Server Run
    def run(self):
        if(self.runevent_faq.load_faq_vectorstore() == True):
            self.registerTools()                                # Register Tool as Runtime.
            self.mcp.run(transport="streamable-http")           # Run Command for MCP Server
        else:
            raise Exception("Error Loading FAQ to Database.")


# Start the main program
if __name__ == "__main__":
    try:
        mcp_server = MCPServer(server_host="localhost", server_port=5002)
        mcp_server.run()
    except KeyboardInterrupt:
        print("\nShutdown signal received. Cleaning up resources...", file=sys.stderr)
        print("Server successfully stopped. Goodbye!", file=sys.stderr)
    except Exception as exp:
        print(f"Shutting down RunEvent MCP Server due to Exception - {exp}", file=sys.stderr)
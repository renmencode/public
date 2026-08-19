# FastMCP Server for User Ticket Management

# Import System Packages 
import sys

# Import FastMCP
from mcp.server.fastmcp import FastMCP

# Import External Modules
from userticket_service import TicketManager


# Create MCP Server
class MCPServer:

    def __init__(self, server_host: str, server_port: int):
        print("MCP Initialization...")
        self.mcp = FastMCP(
            name = "Support_Ticket",
            host = server_host,
            port = server_port,
            stateless_http = True
            )                                               # Initialise FastMCP
        
        self.ticketmanager = TicketManager()                # Initialize Tools

    # Register Tool in MCP Server
    def registerTools(self):
        print("Tool Registration...")
        self.mcp.tool(name="create_user_ticket")(self.ticketmanager.create_user_ticket)
        self.mcp.tool(name="list_user_tickets")(self.ticketmanager.list_user_tickets)
        self.mcp.tool(name="lookup_ticket_status")(self.ticketmanager.lookup_ticket_status)

    # Trigger MCP Server Run
    def run(self):
        self.registerTools()                        # Register Tool as Runtime.
        self.mcp.run(transport="streamable-http")   # Run Command for MCP Server


# Start the main program
if __name__ == "__main__":
    try:
        mcp_server = MCPServer(server_host="localhost", server_port=5000)
        mcp_server.run()
    except KeyboardInterrupt:
        print("\nShutdown signal received. Cleaning up resources...", file=sys.stderr)
        print("Server successfully stopped. Goodbye!", file=sys.stderr)
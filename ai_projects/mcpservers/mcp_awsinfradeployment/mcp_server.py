# FastMCP Server for AWS Infrastructure Deployment

# Import System Packages 
import os, sys
import json
from dotenv import load_dotenv

# Import FastMCP
from mcp.server.fastmcp import FastMCP

# Import External Modules
from awsinfra_artifactdeploy_tool import ArtifactDeploy
#from awsinfra_albdeploy_tool import ALBDeploy
from awsinfra_templatedatasubstitution import TemplateDataSubstitution

# load Env Vars
load_dotenv()

# MCP Server Instruction File. 
mcp_awsinfra_instr_file="C:\\RanjithC\\AIProjects\\PromptEngg\\mcpservers\\data\\awsinfra\\awsinfra_mcp_instructions.json"


# Create MCP Server
class AWSInfraMCPServer:

    def __init__(self):
        print("MCP Initialization...")
        server_host = os.getenv("mcp_awsinfra.host")
        server_port = os.getenv("mcp_awsinfra.port")

        # Initialise FastMCP
        self.mcp = FastMCP(
            name = "AWSInfraMCPServer",
            host = server_host,
            port = server_port,
            stateless_http = True
        )

        # Read MCP Services Instructions
        with open(mcp_awsinfra_instr_file, 'r') as instr:
            self.mcp_awsinfra_instr = json.load(instr)

        # Initialise Tool Services
        self.template_sub = TemplateDataSubstitution(
            self.mcp_awsinfra_instr["template_sub_llm_instr"],
            self.mcp_awsinfra_instr["sub_content_format_instr"],
            self.mcp_awsinfra_instr["ec2_fully_baked_example"]
        )
        self.deploy_artifact = ArtifactDeploy(self.template_sub)
        # self.deploy_alb = ALBDeploy()

    # Register Tool in MCP Server
    def registerTools(self):
        try:
            print("Tool Registration...")
            self.mcp.tool(name="deploy_template")(self.deploy_artifact.deploy)
            self.mcp.tool(name="update_template")(self.deploy_artifact.update_template)

            #self.mcp.tool(name="deploy_alb")(self.deploy_alb.deploy)

        except Exception as exp:
            error = f"Error inside AWSInfraMCPServer.registerTools: {exp}"
            print(error, flush=True)
            raise Exception(error)

    # Trigger MCP Server Run
    def run(self):
        try:
            self.registerTools()
            self.mcp.run(transport="streamable-http")

            #template_values = {'launch_template': {'InstanceType': 't3.large', 'AMIId': 'ami-0fef201115eefe936', 'KeyName': 'MyWebAppEC2KeyPair', 'VPCId': 'vpc-b44953d1', 'SubnetId': 'subnet-2e5c1f77', 'InstanceName': 'EC2-Local', 'EBSVolumeSize': 20, 'EBSVolumeType': 'gp3'}}
            #template_instance = self.deploy_ec2.update_template(template_values, "/Template_EC2.yaml")
            #self.deploy_ec2.deploy("EC2-Local", template_instance)

        except Exception as exp:
            error = f"Error inside AWSInfraMCPServer.run: {exp}"
            print(error, flush=True)
            raise Exception(error)
        

# Start the main program
if __name__ == "__main__":
    try:
        mcp_server = AWSInfraMCPServer()
        mcp_server.run()
    except KeyboardInterrupt:
        print("\nShutdown signal received. Cleaning up resources...", file=sys.stderr)
        print("AWSInfra MCP Server successfully stopped. Goodbye!", file=sys.stderr)
    except Exception as exp:
        print(f"Shutting down AWSInfra MCP Server due to Exception - {exp}", file=sys.stderr)
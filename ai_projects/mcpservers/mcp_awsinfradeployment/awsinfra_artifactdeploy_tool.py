# AWS Infrastucture Artifact Deployment Tool

# Generic Class holds two Tools / Functions for Deployemnt.
#   - update_template, deploy.
# 'update_template' is used to update deployment template.
# 'deploy' is used to deploy the instantiated template.


# Class Late Binding Annotaion
from __future__ import annotations 

# Import Python Native Lib
import os, sys
import json
import uuid as uuid
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict, Literal

# Import AWS boto3 Packages
import boto3
from botocore.exceptions import ClientError

# Load External Class modules
from awsinfra_templatedatasubstitution import TemplateDataSubstitution

# Load Env Vars
load_dotenv()

# AWS Infra Env Attributes 
mcp_awsinfra_templates_repo = os.getenv("mcp_awsinfra_templates")
mcp_awsinfra_cfnprofile = os.getenv("mcp_awsinfra_cfnprofile")
mcp_awsinfra_deployregion = os.getenv("mcp_awsinfra_deployregion")


# Class EC2 Deployment EC2Deploy
class ArtifactDeploy:

    def __init__(self, template_sub: TemplateDataSubstitution):
        self.template_sub = template_sub

    # Deploy EC2 Function
    def deploy(self, deployment_stack_name: str, yaml_template_instance: str) -> dict:
        """This Tool is used to Deploy AWS Artifact Instances using the Fully Baked Launch Templated. 

        Args: 
            deployment_stack_name: Name of Deployment Stack that needs to be Created. (Default - '')
            yaml_template_instance: Fully Baked YAML Launch Template Instance. (Default - '')

        Returns: 
            cfn_stack_output_dict: A Dictionary containing Output Attributes of the Deployment.

        """

        print("\nInside Deployment - InstanceName:\n", deployment_stack_name, flush=True)
        print("\nInside Deployment - Template:\n", yaml_template_instance, flush=True)

        try:
            # Init Output Vars
            cfn_stack_output_dict = {}          # Stack Output Dict
            launch_template_run_output = None   # Flattended JSON String

            # Create CFN Client
            session = boto3.Session(profile_name=mcp_awsinfra_cfnprofile, region_name=mcp_awsinfra_deployregion)
            cfn_client = session.client("cloudformation")
            stack_name = "LHT-" + deployment_stack_name + "-CFT"

            # Deploy EC2 Stack
            cfn_response = cfn_client.create_stack(
                StackName=stack_name,
                TemplateBody=yaml_template_instance,
                Capabilities=['CAPABILITY_IAM', 'CAPABILITY_NAMED_IAM'],
                OnFailure='ROLLBACK'
            )
            cfn_stack_output_dict['StackId'] = cfn_response['StackId']
            cfn_stack_output_dict['Stack_Status'] = "CREATE_IN_PROGRESS"

            print(f"\nStack Run Initiated - Stack ID: {cfn_response['StackId']}")

            # Wait for the Stack Creation to Complete
            waiter = cfn_client.get_waiter('stack_create_complete')
            waiter.wait(StackName=stack_name)

            print(f"\nStack Run Completed - Stack ID: {cfn_response['StackId']}")

            # Update Output Dict with Stack Status & Output Fields.
            cfn_stack_metadata = cfn_client.describe_stacks(StackName=stack_name)
            cfn_stack_data = cfn_stack_metadata['Stacks'][0]
            cfn_stack_output_dict['Stack_Status'] = cfn_stack_data["StackStatus"]

            # Format User defined Stack Outputs - Key/Value
            cfn_stack_outputs = cfn_stack_data.get("Outputs", [])
            if cfn_stack_outputs:
                for out_val in cfn_stack_outputs:
                    key = out_val.get("OutputKey")
                    value = out_val.get("OutputValue")
                    cfn_stack_output_dict[key] = value

            # Flatten Stack Output Dict to JSON String.
            #launch_template_run_output = json.dumps(cfn_stack_output_dict)
            #response = f"Deployment Successful. Output Attributes of Launch Template Run: {launch_template_run_output}"

            print("Output Attributes: ", cfn_stack_output_dict, flush=True)

            return(cfn_stack_output_dict)
    
        except Exception as exp:
            error = f"Error inside ArtifactDeploy.deploy - {exp}"
            print(error, flush=True)
            raise Exception(error)

    # Update Launch Template Function
    def update_template(self, template_parameters: dict | None=None, yaml_launch_template_file: str | None=None) -> str:
        """This tool is used to Update Parameters in Launch Template of AWS Infrastructure. 

        Args: 
            template_parameters: Input Dictionary that contains Launch Parameters and its Values.
                Examples: 
                    - VPC Parameters: {'VpcCidr': '10.0.0.0/16', 'EnvironmentName': 'prod'}
                    - ALB Parameters: {'VpcId': 'VPC12345555', 'SubnetId': 'sub-123456, sub-9876543', 'SecurityGroup': 'sg-12865439'}

            yaml_launch_template_file: Input Launch Template YAML File. File Name pattern as shown in below example.
                Examples: 
                    - ALB Template File: '/Template_ALB.yaml'
                    - ECR Template File: '/Template_ECR.yaml'

        Returns: 
            template_instance: Fully Baked Instance of YAML Launch Template.

        """

        try:
            print("\nInside CloudFormation Update Template\n", flush=True)

            # Read the Contents of EC2 YAML File as Raw.
            ec2_template_file_path = mcp_awsinfra_templates_repo + yaml_launch_template_file.replace('/', '\\')
            with open(ec2_template_file_path, 'r') as ec2_template:
                ec2_template_content = ec2_template.read()

            # Call Template Substitution Class
            template_instance = self.template_sub.do_substitution(template_parameters, ec2_template_content)

            print("\nFully Baked Template:\n", template_instance, flush=True)

            return(template_instance)
    
        except Exception as exp:
            error = f"Error inside ArtifactDeploy.update_template - {exp}"
            print(error, flush=True)
            raise Exception(error)

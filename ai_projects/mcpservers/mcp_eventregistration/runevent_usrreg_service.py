# Class Late Binding Annotaion
from __future__ import annotations 

# Import Python Native Lib
import os, sys
import random
import numpy as np
import uuid as uuid
from dotenv import load_dotenv

from typing_extensions import Annotated, TypedDict, Literal

# Import Custom Modules
from runevent_dbmanager import DBManager

# Load Env Variables
load_dotenv()


# User Schema
class UserInfo(TypedDict):
    first_name: str
    last_name: str
    user_age: int
    user_email: str
    user_phone: str

# User Register Tool
class RunEventUserRegistration:

    def __init__(self):
        self.db_manager = DBManager()
        self.db_manager.create_sqllite_conn()

    # Validate User MCP Tool
    def validate_user(self, user_info: UserInfo) -> str:
        """This tool used to VALIDATE the User Information for the Running Event. 

        Args: 
            user_info: Input Dictionary that will contain User data needed to Validate the User.

        Returns: 
            response: A statement stating whethere the User Validation is Success or Failure. 

        """

        is_validation_failed = False

        # Check for Missing Values
        if (user_info.get("first_name") is None):
            is_validation_failed = True
            response = "User 'First Name' is missing."
        elif (user_info.get("last_name") is None):
            is_validation_failed = True
            response = "User 'Last Name' is missing."
        elif (user_info.get("user_age") is None):
            is_validation_failed = True
            response = "User 'Age' is missing."
        elif (user_info.get("user_email") is None):
            is_validation_failed = True
            response = "User 'Email' is missing."
        elif (user_info.get("user_phone") is None):
            is_validation_failed = True
            response = "User 'Phone Number' is missing."
        else:
            is_validation_failed = False

        # Validate User Age
        if (is_validation_failed == False):
            if (user_info.get("user_age") < 40):
                is_validation_failed = True
                response = "User Age Validation Failed."
            else:
                is_validation_failed = False

        if (is_validation_failed == False):
            response = "User Validation Successful."
        else:
            response = "User Validation Failed." + response

        print("Validate UserInfo: ", response, flush=True)

        return(response)


    # Register User MCP Tool.
    def register_user(self, user_info: UserInfo) -> str:
        """ This tool used to REGISTER a User for the Running Event.

        Args: 
            user_info: Input Dictionary that will contain User data needed to Register the User.

        Returns: 
            response: A statement, stating Successful User Registeration along with the RegisterationID.
            
        """

        cursor = None

        try:
            reg_id = random.randint(100000, 999999)
            first_name = user_info.get("first_name")
            last_name = user_info.get("last_name")
            age = user_info.get("user_age")
            email = user_info.get("user_email")
            phone = user_info.get("user_phone")

            sql_conn = self.db_manager.get_sqllite_conn()
            cursor = sql_conn.cursor()

            cursor.execute("INSERT INTO user_registeration (reg_id, first_name, last_name, age, email, phone_number) VALUES (?, ?, ?, ?, ?, ?)", 
                           (reg_id, first_name, last_name, age, email, phone)
                        )            
            sql_conn.commit()

            response = f"User Registration Successful. Registeration No# - {reg_id}"

            print(response, flush=True)
            
            return(response)
        
        except Exception as exp:
            error = f"Error inside RunEventUserRegistration.register_user - {exp}"
            print(error, flush=True)
            raise Exception(error)
        finally:
            if (cursor is not None):
                cursor.close()


    # Lookup Registered User for the Event.
    def lookup_registered_user(self, regId: int) -> str:
        """ This Tool is used to Lookup a specific Registered User for the Running Event.

        Args:
            regId: RegisterationID of the User, that is used to do the lookup in the database. 

        Returns:
            response: Information pertaining to Registered Users in the format - {first_name, last_name, age, email, phone_number}

        """
        
        cursor = None

        try:
            sql_conn = self.db_manager.get_sqllite_conn()
            cursor = sql_conn.cursor()
            cursor.execute("SELECT first_name, last_name, age, email, phone_number FROM user_registeration WHERE reg_id = ?",
                          (regId,)
                        )

            record = cursor.fetchall()
            if (record == None):
                response = f"No User found with this RegisterationID - {regId}."
            else:
                record_list = [row for row in record]
                user_info = {
                    "first_name": record_list[0][0],
                    "last_name": record_list[0][1],
                    "age": record_list[0][2],
                    "email": record_list[0][3],
                    "phone_number": record_list[0][4]
                }
                response = f"Registered User - {user_info}"

            print(response, flush=True)

            return(response)

        except Exception as exp:
            error = f"Error inside RunEventUserRegistration.lookup_registered_user - {exp}"
            print(error, flush=True)
            raise Exception(error)
        finally:
            if (cursor is not None):
                cursor.close()


    # List of Users Registered for the Event.
    def list_registered_users(self) -> str:
        """ This Tool is used to Lookup list of Users for the Running Event. 

        Returns:
            response: List of Registered Users.

        """
        
        cursor = None

        try:
            sql_conn = self.db_manager.get_sqllite_conn()
            cursor = sql_conn.cursor()
            cursor.execute("SELECT * FROM user_registeration")

            record = cursor.fetchall()
            if (record == None):
                response = f"No Users have been Registered."
            else:
                record_list = [row for row in record]
                response = f"List of Registered Users - {record_list}"

            print(response, flush=True)

            return(response)

        except Exception as exp:
            error = f"Error inside RunEventUserRegistration.list_registered_users - {exp}"
            print(error, flush=True)
            raise Exception(error)
        finally:
            if (cursor is not None):
                cursor.close()

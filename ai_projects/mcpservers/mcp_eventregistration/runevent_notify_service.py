# Class Late Binding Annotaion
from __future__ import annotations 

# Import Python Native Lib
import os, sys
import asyncio
import numpy as np
import uuid as uuid
from dotenv import load_dotenv

from typing_extensions import Annotated, TypedDict, Literal

# Load Env Variables
load_dotenv()


class NotifyInfo(TypedDict):
    notify_msg: str
    user_email: str


class RunEventNotifyUser:

    def __init__(self):
        pass

    def notify_user(self, notify_info: NotifyInfo):
        """This tool is used to NOTIFY Users if the Registeration for the Run Event was Success of Failure.

        Args:
            notify_info: Input Dictionary that will Notification Message and Users EMail.

        Returns:
            response: A statement stating whether Notification was Successful or Failure.

        Notification Message Format if Registration Process was a SUCCESS:

            Dear {User First Name} {User's Last Name},

            Congatulations on your successful registration for the Running Event.
            Your Registeration ID for the Event is - {RegisterationID}.

            Regards,
            RunFun Committee

        Notification Message Format if Registration Process was a FAILURE:

            Dear {User First Name} {User's Last Name},

            Sorry, your Registration Process Failed. {Specify the Reason for Failure here}

            Regards,
            RunFun Committee

        """

        print("Notify Message: ", notify_info.get("notify_msg"), flush=True)

        print("User EMail: ", notify_info.get("user_email"), flush=True)

        return("Notification Successful.")
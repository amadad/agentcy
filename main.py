import os
import openai
from autogen import AssistantAgent, UserProxyAgent
from dotenv import load_dotenv

load_dotenv()

assistant = AssistantAgent("assistant")
user_proxy = UserProxyAgent("user_proxy")
user_proxy.initiate_chat(assistant, message="Find the value proposition for any brand. Ask me for input.")
# This initiates an automated chat between the two agents to solve the task
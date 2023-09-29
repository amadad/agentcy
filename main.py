import os
import openai
import autogen
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")

user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message="A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.",
    code_execution_config=False,
)

fun_engineer = autogen.AssistantAgent(
    name="Fun_Manager",
    llm_config={"config_list": config_list},
    system_message='''Fun Manager. You maximize the fun when Admin is at a location - optimize for unique memorable experiences & fun stories.'''
)

gym_trainer = autogen.AssistantAgent(
    name="Gym_Trainer",
    llm_config={"config_list": config_list},
    system_message="Gym Trainer. You make sure admin is getting the right training (lifting 4-5 times a week) and eating the right food to get to a 6-pack."
)

executive_assistant = autogen.AssistantAgent(
    name="Executive_Assistant",
    llm_config={"config_list": config_list},
    system_message="Executive Assistant. You make sure the daily work (like project deadlines & daily habits like design and copywriting practice) required by the Admin is done before any of the fun activities."
)

planner = autogen.AssistantAgent(
    name="Planner",
    llm_config={"config_list": config_list},
    system_message='''Planner. Suggest a plan. Revise the plan based on feedback from admin, Executive Assistant, Fun Manager, until admin approval. Explain the plan first. Be clear which step is performed by an engineer, and which step is performed by a scientist.'''
)

critic = autogen.AssistantAgent(
    name="Critic",
    llm_config={"config_list": config_list},
    system_message="Critic. Double check plan, make sure all objectives from fun manager, executive assistant, and gym trainer are met. Provide feedback."
)

groupchat = autogen.GroupChat(agents=[
    user_proxy, fun_engineer, executive_assistant, gym_trainer, planner, critic], messages=[], max_round=3)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})

user_proxy.initiate_chat(
    manager,
    message="""
    Plan a month-long trip to Bangkok. Include a table of dates and activity. This will give you a list of tasks that need to be done on a particular day.
    """,
)

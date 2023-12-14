import os
import requests
from bs4 import BeautifulSoup
import json
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from dotenv import load_dotenv
# Import memgpt related modules
import memgpt.autogen.memgpt_agent as memgpt_autogen
import memgpt.autogen.interface as autogen_interface 
import memgpt.presets as presets
from memgpt.persistence_manager import InMemoryStateManager

# Load environment variables
load_dotenv()

# Existing configuration
config_list = autogen.config_list_from_dotenv(
    dotenv_file_path='.env',
    filter_dict={"model": {"gpt-4-1106-preview", "gpt-4"}}
)
llm_config = {"config_list": config_list}

llm_config = {"config_list": config_list, "seed": 42}
user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    code_execution_config={"last_n_messages": 2, "work_dir": "groupchat"},
)

interface = autogen_interface.AutoGenInterface() # how MemGPT talks to AutoGen
persistence_manager = InMemoryStateManager()
persona = "I\'m a 10x engineer at a FAANG tech company."
human = "I\'m a team manager at a FAANG tech company."
memgpt_agent = presets.use_preset(presets.DEFAULT, 'gpt-4', persona, human, interface, persistence_manager)

# MemGPT coder
coder = memgpt_autogen.MemGPTAgent(
    name="MemGPT_coder",
    agent=memgpt_agent,
)

# non-MemGPT PM
pm = autogen.AssistantAgent(
    name="Product_manager",
    system_message="Creative in software product ideas.",
    llm_config=llm_config,
)

groupchat = autogen.GroupChat(agents=[user_proxy, coder, pm], messages=[], max_round=12)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

user_proxy.initiate_chat(manager, message="First send the message 'Let's go Mario!'")
import autogen
from autogen import config_list_from_json, UserProxyAgent, AssistantAgent, GroupChat, GroupChatManager
from autogen.agentchat.contrib.agent_builder import AgentBuilder 

# 1. Configuration
config_path = 'OAI_CONFIG_LIST.json'
config_list = config_list_from_json("OAI_CONFIG_LIST")
default_llm_config = {'temperature': 0}

# 2. User inputs
brand_task = input("Please enter the brand or company name: ")
building_task = input("Please enter the your goal, brief, or problem statement: ")
termination_notice = (
    '\n\nDo not show appreciation in your responses, say only what is necessary. '
    'if "Thank you" or "You\'re welcome" are said in the conversation, then say TERMINATE '
    'to indicate the conversation is finished and this is your last message.'
)

# 3. Initialising Builder
builder = AgentBuilder(config_path=config_path)
agent_list, agent_configs = builder.build(building_task, default_llm_config)


group_chat = autogen.GroupChat(agents=agent_list, messages=[], max_round=12)
manager = autogen.GroupChatManager(groupchat=group_chat, llm_config={"config_list": config_list, **default_llm_config})
agent_list[0].initiate_chat(
    manager, 
    message=building_task,
)

import os
import requests
from bs4 import BeautifulSoup
import json
import autogen
import openai
from autogen import config_list_from_json, UserProxyAgent, AssistantAgent, GroupChat, GroupChatManager
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate 
from dotenv import load_dotenv
from tools import search, scrape, summary, research, write_content, save_markdown

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
BROWSERLESS_API_KEY = os.getenv("BROWSERLESS_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
config_list = config_list_from_json("OAI_CONFIG_LIST")

brand_task = input("Please enter the brand or company name: ")
user_task = input("Please enter the your goal, brief, or problem statement: ")

llm_config_content_assistant = {
    "functions": [
        {
            "name": "research",
            "description": "research about a given topic, return the research material including reference links",
            "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The topic to be researched about",
                        }
                    },
                "required": ["query"],
            },
        },
        {
            "name": "write_content",
            "description": "Write content based on the given research material & topic",
            "parameters": {
                    "type": "object",
                    "properties": {
                        "research_material": {
                            "type": "string",
                            "description": "research material of a given topic, including reference links when available",
                        },
                        "topic": {
                            "type": "string",
                            "description": "The topic of the content",
                        }
                    },
                "required": ["research_material", "topic"],
            },
        },
    ],
    "config_list": config_list}

agency_manager = AssistantAgent(
    name="Agency_Manager",
    description="Outlines plan for agents.",
    llm_config={"config_list": config_list},
    system_message=f'''
    You are the Project Manager, focusing on {brand_task}. 
    Outline step-by-step tasks for {user_task} with the team. 
    Act as a communication hub, maintain high-quality deliverables, and regularly update all stakeholders on progress. 
    Terminate the conversation with "TERMINATE" when all tasks are completed and no further actions are needed.
    '''
)

agency_researcher = AssistantAgent(
    name="Agency_Researcher",
    description="Conducts detailed research to provide insights and information vital for executing user-focused tasks.",
    llm_config=llm_config_content_assistant,
    system_message=f'''
    As the Lead Researcher, your primary role revolves around {user_task}. 
    Utilize the research function to gather in-depth insights about market trends, user pain points, and cultural dynamics relevant to our project. 
    Provide these insights proactively to support the team's strategy and decision-making. In your responses, focus on delivering clear, actionable information. 
    Conclude your participation with "TERMINATE" once all relevant research has been provided and no further analysis is required.
    ''',
    function_map={
        "research": research,
    }
)

agency_researcher.register_function(
    function_map={
        "research": research,
    }
)

agency_strategist = AssistantAgent(
    name="Agency_Strategist",
    description="Develops strategic briefs based on market analysis and research findings, focusing on brand positioning and audience insights.",
    llm_config={"config_list": config_list},
    system_message=f'''
    As the Lead Strategist, your key task is to develop strategic briefs for {brand_task}, guided by {user_task} objectives. 
    Utilize the insights from Agency_Researcher to inform your strategies, focusing on brand positioning, key messaging, and audience targeting. 
    Ensure your briefs offer unique perspectives and clear direction. 
    Coordinate closely with the Agency_Manager for alignment with project goals. 
    Conclude with "TERMINATE" once the strategic direction is established and communicated.
    '''
)

agency_writer = AssistantAgent(
    name="Agency_Copywriter",
    description="Creates engaging content and narratives aligned with project goals, using insights from research and strategy.",
    llm_config={"config_list": config_list},
    system_message=f'''
    As the Lead Copywriter, your role is to craft compelling narratives and content.
    Focus on delivering clear, engaging, and relevant messages that resonate with our target audience. 
    Use your creativity to transform strategic insights and research findings into impactful content. 
    Ensure your writing maintains the brand's voice and aligns with the overall project strategy. 
    Your goal is to create content that effectively communicates our message and engages the audience.
    ''',
    function_map={
        "write_content": write_content,
    }
)

writing_assistant = AssistantAgent(
    name="writing_assistant",
    description="Versatile assistant skilled in researching various topics and crafting well-written content.",
    llm_config=llm_config_content_assistant,
    system_message=f'''
    As a writing assistant, your role involves using the research function to stay updated on diverse topics and employing the write_content function to produce polished prose. 
    Ensure your written material is informative and well-structured, catering to the specific needs of the topic. 
    Conclude your contributions with "TERMINATE" after completing the writing tasks as required.
    ''',
    function_map={
        "research": research,
        "write_content": write_content,
    }
)

agency_marketer = AssistantAgent(
    name="Agency_Marketer",
    description="Crafts marketing strategies and campaigns attuned to audience needs, utilizing insights from project research and strategy.",
    llm_config={"config_list": config_list},
    system_message=f'''
    As the Lead Marketer, utilize insights and strategies to develop marketing ideas that engage our target audience. 
    For {user_task}, create campaigns and initiatives that clearly convey our brand's value. 
    Bridge strategy and execution, ensuring our message is impactful and aligned with our vision. 
    Collaborate with teams for a unified approach, and coordinate with the Agency Manager for project alignment. 
    Conclude with "TERMINATE" when your marketing contributions are complete.
    '''
)

agency_mediaplanner = AssistantAgent(
    name="Agency_Media_Planner",
    description="Identifies optimal media channels and strategies for ad delivery, aligned with project goals.",
    llm_config={"config_list": config_list},
    system_message=f'''
    As the Lead Media Planner, your task is to identify the ideal media mix for delivering our advertising messages, targeting the client's audience. 
    Utilize the research function to stay updated on current and effective media channels and tactics. 
    Apply insights from {user_task} to formulate strategies that effectively reach the audience through various media, both traditional and digital. 
    Collaborate closely with the Agency Manager to ensure your plans are in sync with the broader user strategy. 
    Conclude your role with "TERMINATE" once the media planning is complete and aligned.
    '''
)

agency_director = AssistantAgent(
    name="Agency_Director",
    description="Guides the project's creative vision, ensuring uniqueness, excellence, and relevance in all ideas and executions.",
    llm_config={"config_list": config_list},
    system_message=f'''
    As the Creative Director, your role is to oversee the project's creative aspects. 
    Critically evaluate all work, ensuring each idea is not just unique but also aligns with our standards of excellence. 
    Encourage the team to innovate and explore new creative avenues. 
    Collaborate closely with the Agency_Manager for consistent alignment with the user_proxy. 
    Conclude your guidance with "TERMINATE" once you've ensured the project's creative integrity and alignment.
    '''
)

user_proxy = UserProxyAgent(
   name="user_proxy",
   description="Acts as a proxy for the user, capable of executing code and handling user interactions within predefined guidelines.",
   is_termination_msg=lambda msg: "TERMINATE" in msg["content"] if msg["content"] else False,
   human_input_mode="TERMINATE",
   max_consecutive_auto_reply=1,
   code_execution_config={"work_dir": "/logs"},
   system_message='Be a helpful assistant.',
)

groupchat = GroupChat(agents=[
    user_proxy, agency_manager, agency_researcher, agency_strategist, agency_writer, writing_assistant, agency_marketer, agency_mediaplanner, agency_director], 
    messages=[], 
    max_round=20
)

manager = GroupChatManager(
    groupchat=groupchat, 
    llm_config={"config_list": config_list}
)

user_proxy.initiate_chat(
    manager, 
    message=user_task,
)
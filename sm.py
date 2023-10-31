import os
import requests
from bs4 import BeautifulSoup
import json
import autogen
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate 
import openai
from dotenv import load_dotenv
import langchain.globals

# Load environment variables
load_dotenv()
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
BROWSERLESS_API_KEY = os.getenv("BROWSERLESS_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

autogen.ChatCompletion.start_logging()
brand_task = input("Please enter the brand or company name: ")
user_task = input("Please enter the your goal, brief, or problem statement: ")
problem_task = ""

my_cache_value = {...} 
langchain.globals.set_llm_cache(my_cache_value)
cache = langchain.globals.get_llm_cache()

def search(query):
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query
    })
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json()

def scrape(url: str):
    """Scrape a website and summarize its content if it's too large."""
    print("Scraping website...")
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }
    post_url = f"https://chrome.browserless.io/content?token={BROWSERLESS_API_KEY}"
    response = requests.post(post_url, headers=headers, json={"url": url})
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENTTTTTT:", text)
        if len(text) > 8000:
            output = summary(text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")

def summary(content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a detailed summary of the following text for a research purpose:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text"])
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )
    output = summary_chain.run(input_documents=docs,)
    return output

def research(query):
    llm_config_researcher = {
        "functions": [
            {
                "name": "search",
                "description": "google search for relevant information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Google search query",
                        }
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "scrape",
                "description": "Scraping website content based on url",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Website url to scrape",
                        }
                    },
                    "required": ["url"],
                },
            },
        ],
        "config_list": config_list}

    research_assistant = autogen.AssistantAgent(
        name="research_assistant",
        system_message=f'''
        Welcome, Research Assistant.
        Your task is to research the provided query extensively. 
        Produce a detailed report, ensuring you include technical specifics and reference all sources. Conclude your report with "TERMINATE".
        ''',
        llm_config=llm_config_researcher,
    )

    user_proxy = autogen.UserProxyAgent(
        name="User_proxy",
        code_execution_config={"last_n_messages": 2, "work_dir": "coding"},
        is_termination_msg=lambda x: x.get("content", "") and x.get(
            "content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="TERMINATE",
        function_map={
            "search": search,
            "scrape": scrape,
        }
    )

    user_proxy.initiate_chat(research_assistant, message=query)
    user_proxy.stop_reply_at_receive(research_assistant)
    user_proxy.send(
        "Give me the research report that just generated again, return ONLY the report & reference links.", research_assistant)
    return user_proxy.last_message()["content"]

def write_content(research_material, topic):
    editor = autogen.AssistantAgent(
        name="editor",
        system_message=f'''
        Welcome, Senior Editor.
        As a seasoned professional, you bring meticulous attention to detail, a deep appreciation for literary and cultural nuance, and a commitment to upholding the highest editorial standards. 
        Your role is to craft the structure of a short blog post using the material from the Research Assistant. Use your experience to ensure clarity, coherence, and precision. 
        Once structured, pass it to the Writer to pen the final piece.
        ''',
        llm_config={"config_list": config_list},
    )

    writer = autogen.AssistantAgent(
        name="writer",
        system_message=f'''
        Welcome, Blogger.
        Your task is to compose a short blog post using the structure given by the Editor and incorporating feedback from the Reviewer. 
        Embrace stylistic minimalism: be clear, concise, and direct. 
        Approach the topic from a journalistic perspective; aim to inform and engage the readers without adopting a sales-oriented tone. 
        After two rounds of revisions, conclude your post with "TERMINATE".
        ''',
        llm_config={"config_list": config_list},
    )

    reviewer = autogen.AssistantAgent(
        name="reviewer",
        system_message=f'''
        As a distinguished blog content critic, you are known for your discerning eye, deep literary and cultural understanding, and an unwavering commitment to editorial excellence. 
        Your role is to meticulously review and critique the written blog, ensuring it meets the highest standards of clarity, coherence, and precision. 
        Provide invaluable feedback to the Writer to elevate the piece. After two rounds of content iteration, conclude with "TERMINATE".
        ''',        
        llm_config={"config_list": config_list},
    )

    user_proxy = autogen.UserProxyAgent(
        name="admin",
        system_message="A human admin. Interact with editor to discuss the structure. Actual writing needs to be approved by this admin.",
        code_execution_config=False,
        is_termination_msg=lambda x: x.get("content", "") and x.get(
            "content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="TERMINATE",
    )

    groupchat2 = autogen.GroupChat(
        agents=[user_proxy, editor, writer, reviewer],
        messages=[],
        max_round=10)
    
    manager = autogen.GroupChatManager(groupchat=groupchat2)

    user_proxy.initiate_chat(
        manager, message=f"Write a blog about {topic}, here are the material: {research_material}")

    user_proxy.stop_reply_at_receive(manager)
    user_proxy.send(
        "Give me the blog that just generated again, return ONLY the blog, and add TERMINATE in the end of the message", manager)
    return user_proxy.last_message()["content"]

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

agency_manager = autogen.AssistantAgent(
    name="Agency_Manager",
    llm_config={"config_list": config_list},
    system_message=f'''
    You are the Project Manager. Be concise and refrain from any conversations that don't serve the goal of the user, ie. thank you.
    Rewrite and reframe the {user_task} for {brand_task} as {problem_task}.
    Think step by step. Your primary responsibility is to oversee the entire project lifecycle, ensuring that all agents are effectively fulfilling their objectives.
    Reply TERMINATE when your task is done.
    '''
)

agency_strategist = autogen.AssistantAgent(
    name="Agency_Strategist",
    llm_config={"config_list": config_list},
    system_message=f'''
    You are the Lead Strategist. Be concise and refrain from any conversations that don't serve the goal of the user, ie. thank you.
    Draft a strategic brief that effectively positions {brand_task} for {problem_task}.
    Think step by step. Use the research function to search to delevop {brand_task}'s unique value proposition, target audience, and competitive landscape. 
    Reply TERMINATE when your task is done.
    ''',
    function_map={
        "research": research
    }
)

agency_researcher = autogen.AssistantAgent(
    name="Agency_Researcher",
    llm_config=llm_config_content_assistant,
    system_message=f'''
    You are the Lead Researcher, you can use the research function to search to delevop {problem_task}'s user pain points, identifying market opportunities, and analyzing prevailing market conditions. 
    Be concise and refrain from any conversations that don't serve the goal of the user, ie. thank you.
    Draft a research report that effectively contextualizes {brand_task} for {problem_task}.
    Think step by step. 
    Reply TERMINATE when your task is done.
    ''',
    function_map={
        "research": research
    }
)

agency_writer = autogen.AssistantAgent(
    name="Agency_Copywriter",
    llm_config={"config_list": config_list},
    system_message=f'''
    You are the Lead Copywriter, you can use research function to collect latest information about {brand_task} and {problem_task}.
    Be concise and refrain from any conversations that don't serve the goal of the user, ie. thank you.
    Craft a compelling narrative framework and message map that align with the {brand_task}'s strategy and resonate with its audience. 
    Use appropriate persuasive principles: Reciprocity, Scarcity, Authority, Commitment, consistency, Consensus/Social proof, Liking.
    Reply TERMINATE when your task is done.
    ''',
    function_map={
        "write_content": write_content
    }
)

writing_assistant = autogen.AssistantAgent(
    name="writing_assistant",
    llm_config=llm_config_content_assistant,
    system_message=f'''
    You are a writing assistant, you can use research function to collect latest information about a given topic, 
    and then use write_content function to write  persuasive copy using principles: Reciprocity, Scarcity, Authority, Commitment, consistency, Consensus/Social proof, Liking.
    Reply TERMINATE when your task is done
    ''',
    function_map={
        "research": research
    }
)

agency_marketer = autogen.AssistantAgent(
    name="Agency_Marketer",
    llm_config={"config_list": config_list},
    system_message=f'''
    You are the Lead Marketer. Be concise and refrain from any conversations that don't serve the goal of the user, ie. thank you.
    Select an appropriate marketing framework: 4P's of Marketing, STP (Segmentation, Targeting, Positioning), Ansoff Matrix, AIDA (Attention, Interest, Desire, Action), Customer Journey Mapping, Marketing Funnel.
    Identify compelling ideas to deliver an advertising message for {brand_task} and {problem_task}.
    Reply TERMINATE when your task is done    
    '''
)

agency_mediaplanner = autogen.AssistantAgent(
    name="Agency_Media_Planner",
    llm_config={"config_list": config_list},
    system_message=f'''
    You are the Lead Marketer. Be concise and refrain from any conversations that don't serve the goal of the user, ie. thank you.
    Select an appropriate marketing framework: RACE (Reach, Act, Convert, Engage) Framework; See, Think, Do, Care (by Google); AIDA (Attention, Interest, Desire, Action); POEM (Paid, Owned, Earned Media); OST (Objectives, Strategy, Tactics):
    Identify the best mix of media channels to deliver an advertising message for {brand_task} and {problem_task}.
    Reply TERMINATE when your task is done    
    '''
)

user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    human_input_mode="TERMINATE",
    function_map={
        "write_content": write_content,
        "research": research,
    }
)

user_proxy.register_function(
    function_map={
        "research": research,
        "write_content": write_content
    }
)

groupchat = autogen.GroupChat(agents=[
    user_proxy, agency_manager, agency_researcher, agency_strategist, agency_mediaplanner, agency_writer, writing_assistant, agency_marketer], messages=[], max_round=20)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})

user_proxy.initiate_chat(
    manager, 
    message=f"""
    {user_task}
    """,
)
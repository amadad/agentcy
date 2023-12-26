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

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
BROWSERLESS_API_KEY = os.getenv("BROWSERLESS_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
config_list = config_list_from_json("OAI_CONFIG_LIST")

brand_task = input("Please enter the brand or company name: ")
user_task = input("Please enter the your goal, brief, or problem statement: ")
termination_notice = (
    '\n\nDo not show appreciation in your responses, say only what is necessary. '
    'if "Thank you" or "You\'re welcome" are said in the conversation, then say TERMINATE '
    'to indicate the conversation is finished and this is your last message.'
)

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
    
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }
    
    # Build the POST URL
    post_url = f"https://chrome.browserless.io/content?token={BROWSERLESS_API_KEY}"
    
    # Send the POST request
    response = requests.post(post_url, headers=headers, json={"url": url})

    # Check the response status code
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
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")
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

    researcher = AssistantAgent(
        name="researcher",
        system_message="Research about a given query, collect as many information as possible, and generate detailed research results with loads of technique details with all reference links attached; Add TERMINATE to the end of the research report;",
        llm_config=llm_config_researcher,
    )

    user_proxy = UserProxyAgent(
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

    user_proxy.initiate_chat(researcher, message=query)
    user_proxy.stop_reply_at_receive(researcher)
    user_proxy.send(
        "Give me the research report that just generated again, return ONLY the report & reference links", researcher)

    # return the last message the expert received
    return user_proxy.last_message()["content"]

def write_content(research_material, topic):
    editor = autogen.AssistantAgent(
        name="editor",
        system_message='''
        Welcome, Senior Editor.
        As a seasoned professional, you bring meticulous attention to detail, a deep appreciation for literary and cultural nuance, and a commitment to upholding the highest editorial standards. 
        Your role is to craft the structure of a short blog post using the material from the Research Assistant. Use your experience to ensure clarity, coherence, and precision. 
        Once structured, pass it to the Writer to pen the final piece.
        ''',
        llm_config={"config_list": config_list},
    )

    writer = autogen.AssistantAgent(
        name="writer",
        system_message='''
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
        system_message='''
        As a distinguished blog content critic, you are known for your discerning eye, deep literary and cultural understanding, and an unwavering commitment to editorial excellence. 
        Your role is to meticulously review and critique the written blog, ensuring it meets the highest standards of clarity, coherence, and precision. 
        Provide invaluable feedback to the Writer to elevate the piece. After two rounds of content iteration, conclude with "TERMINATE".
        ''',        
        llm_config={"config_list": config_list},
    )

    user_proxy = UserProxyAgent(
        name="admin",
        system_message="A human admin. Interact with editor to discuss the structure. Actual writing needs to be approved by this admin.",
        code_execution_config=False,
        is_termination_msg=lambda x: x.get("content", "") and x.get(
            "content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="TERMINATE",
    )

    groupchat2 = GroupChat(
        agents=[user_proxy, editor, writer, reviewer],
        messages=[],
        max_round=10)
    
    manager = GroupChatManager(groupchat=groupchat2)

    user_proxy.initiate_chat(
        manager, message=f"Write a blog about {topic}, here are the material: {research_material}")

    user_proxy.stop_reply_at_receive(manager)
    user_proxy.send(
        "Give me the blog that just generated again, return ONLY the blog, and add TERMINATE in the end of the message", manager)

    # return the last message the expert received
    return user_proxy.last_message()["content"]

def save_markdown(conversation_log):
    with open("/logs/conversation_log.md", "a") as file:
        file.write(conversation_log + "\n")

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
    llm_config={"config_list": config_list},
    system_message=f'''
    You are the Project Manager. Think about {user_task} step by step for {brand_task} and coordinate with all involved agents.
    ALL questions MUST be answered by appropriate agents. Ask each agent to follow up.
    Ensure that user feedback is promptly incorporated, and any adjustments are made in real-time to align with the project's goals.
    Act as the central point of communication, facilitating collaboration between teams and ensuring that all deliverables are of the highest quality. 
    Regularly review the project's status, address any challenges, and ensure that all stakeholders are kept informed of the project's progress.
    Coordinate with the Agency Director for periodic reviews and approvals, ensuring that the project aligns with the creative vision.
    See {termination_notice}.
    '''
)

agency_researcher = AssistantAgent(
    name="Agency_Researcher",
    llm_config={"config_list": config_list},
    system_message=f'''
    You are the Lead Researcher. You MUST Use the research function to {user_task}.
    Delve deep into understanding user pain points, identifying market opportunities, and analyzing prevailing cultural conditions.
    See {termination_notice}.
    ''',
    function_map={
        "research": research,
    }
)

agency_strategist = AssistantAgent(
    name="Agency_Strategist",
    llm_config={"config_list": config_list},
    system_message=f'''
    You are the Lead Strategist. You MUST use the results of Agency_Researcher in your process.
    Draft a strategic brief that effectively position our {brand_task} and addressess {user_task}.
    Outlines the brand's positioning, key messages, unique value proposition, target audience, competitive landscape, strategic initiatives.
    You MUST use the "research" function to find realtime information about topics you are unable to answer.
    Ensure that the brief contains uncommon insights that unlock a clear understanding.
    Work in tandem with the Agency Manager to ensure alignment with the user_proxy.
    See {termination_notice}.
    ''',
    function_map={
        "research": research,
    }
)

agency_writer = AssistantAgent(
    name="Agency_Copywriter",
    llm_config={"config_list": config_list},
    system_message=f'''
    You are the Lead Copywriter. You MUST use the results from Agency_Researcher and Agency_Strategist in your response.
    Craft compelling narratives and messages that align with {user_task} for {brand_task}.
    Create differentiated content, impactful headlines, in-depth articles, ensuring that the brand's voice is consistent and compelling.
    Work in tandem with the Agency Manager to ensure alignment with the user_proxy.
    See {termination_notice}.
    ''',
    function_map={
        "write_content": write_content,
    }
)

writing_assistant = AssistantAgent(
    name="writing_assistant",
    llm_config=llm_config_content_assistant,
    system_message=f'''
    You are a writing assistant. You MUST use research function to collect the latest information about a given topic, 
    and then use write_content function to write a pristine prose;
    See {termination_notice}.
    ''',
    function_map={
        "research": research,
    }
)

agency_marketer = AssistantAgent(
    name="Agency_Marketer",
    llm_config={"config_list": config_list},
    system_message=f'''
    You are the Lead Marketer. 
    Take the strategy and insights derived from research and transform them into compelling marketable ideas that resonate with the target audience.
    Using the strategic direction from {user_task}, craft innovative marketing campaigns, promotions, and initiatives that effectively communicate the brand's value proposition.
    Your expertise will bridge the gap between strategy and execution, ensuring that the brand's message is not only clear but also captivating. It's essential that your ideas are both impactful and aligned with the brand's overall vision.
    Collaborate with other teams to ensure a cohesive approach, and always strive to push the boundaries of creativity to set our client's brand apart in the market.
    Work in tandem with the Agency Manager to ensure alignment with the user_proxy.
    See {termination_notice}.
    '''
)

agency_mediaplanner = AssistantAgent(
    name="Agency_Media_Planner",
    llm_config={"config_list": config_list},
    system_message=f'''
    You are the Lead Media Planner. 
    Identify the best mix of media channels to deliver an advertising message to a clients' target audience.
    You MUST use the research function to query about contemporary and effective channels and tactics.
    Using insights from {user_task}, strategize the most effective way to get a brand's message to its audience, whether it be through traditional media, digital platforms, or a combination of both.
    Work in tandem with the Agency Manager to ensure alignment with the user_proxy.
    See {termination_notice}.
    ''',
    function_map={
        "research": research,
    }
)

agency_director = AssistantAgent(
    name="Agency_Director",
    llm_config={"config_list": config_list},
    system_message=f'''
    You are the Creative Director at SCTY. 
    Critique and interrogate all work produced to ensure that all ideas are not only unique and compelling but also meet the highest standards of excellence and desirability.
    Challenge the team to think outside the box and push the boundaries of creativity.
    Work in tandem with the Agency Manager to ensure alignment with the user_proxy.
    See {termination_notice}.
    '''
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"] if msg["content"] else False,
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=1,
    code_execution_config={"work_dir": "/logs"},
    system_message="Use the research and write_content functions as needed to answer the user's questions.",
    function_map={
        "research": research,
        "write_content": write_content,
        "save_markdown": save_markdown,
    }
)

user_proxy.register_function(
    function_map={
        "research": research,
        "write_content": write_content,
    }
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
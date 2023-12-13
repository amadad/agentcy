import os
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from bs4 import BeautifulSoup
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import json
from autogen import config_list_from_json
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from autogen import UserProxyAgent
import autogen

load_dotenv()
#config_list = autogen.config_list_from_dotenv(dotenv_file_path='.env')
BROWSERLESS_API_KEY = os.getenv("BROWSERLESS_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
config_list = config_list_from_json("OAI_CONFIG_LIST")

user_task = input("Please enter the your goal, brief, or problem statement: ")

# ------------------ Create functions ------------------ #

# Function for google search
def google_search(search_keyword):    
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": search_keyword
    })

    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    print("RESPONSE:", response.text)
    return response.text

# Function for summarizing
def summary(objective, content):
    llm = ChatOpenAI(temperature = 0, model = "gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size = 10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "objective"])
    
    summary_chain = load_summarize_chain(
        llm=llm, 
        chain_type='map_reduce',
        map_prompt = map_prompt_template,
        combine_prompt = map_prompt_template,
        verbose = False
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output

# Function for scraping
def web_scraping(objective: str, url: str):
    print("Scraping website...")
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    data = {
        "url": url        
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    response = requests.post(f"https://chrome.browserless.io/content?token={BROWSERLESS_API_KEY}", headers=headers, data=data_json)
    
    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENTTTTTT:", text)
        if len(text) > 10000:
            output = summary(objective,text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")    

        # ------------------ Create agent ------------------ #

# Create user proxy agent
user_proxy = UserProxyAgent(name="user_proxy",
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=1
    )

# Create researcher agent
researcher = GPTAssistantAgent(
    name = "researcher",
    llm_config = {
        "config_list": config_list,
        "assistant_id": "asst_cHeIz4xErrE6gIaCDaJU3JJ1"
    }
)

researcher.register_function(
    function_map={
        "web_scraping": web_scraping,
        "google_search": google_search
    }
)

# Create research manager agent
research_manager = GPTAssistantAgent(
    name="research_manager",
    llm_config = {
        "config_list": config_list,
        "assistant_id": "asst_MlgKumQSJluKz1LfFiG9CV4e"
    }
)

# Create director agent
director = GPTAssistantAgent(
    name = "director",
    llm_config = {
        "config_list": config_list,
        "assistant_id": "asst_f8e01JTAXfzSTre5Ffz6lcON",
    }
)

# Create group chat
groupchat = autogen.GroupChat(agents=[user_proxy, researcher, research_manager, director], messages=[], max_round=15)
group_chat_manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})

# ------------------ start conversation ------------------ #

user_proxy.initiate_chat(
    group_chat_manager, 
    message=user_task,
)
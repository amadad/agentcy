import os
import time
import json
import requests
import openai
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import autogen
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate 

# Load environment variables and start logging
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
BROWSERLESS_API_KEY = os.getenv("BROWSERLESS_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")

autogen.ChatCompletion.start_logging()

# Prompt the user for input
user_task = input("Please enter your brand brief: ")
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")

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
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    response = requests.post(
        "https://chrome.browserless.io/content?token=2db344e9-a08a-4179-8f48-195a2f7ea6ee", headers=headers, data=data_json)

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

    researcher = autogen.AssistantAgent(
        name="researcher",
        system_message="Research about a given query, collect as many information as possible, and generate detailed research results with loads of technique details with all reference links attached; Add TERMINATE to the end of the research report;",
        llm_config=llm_config_researcher,
    )

    client = autogen.UserProxyAgent(
        name="client",
        code_execution_config={"last_n_messages": 2, "work_dir": "coding"},
        is_termination_msg=lambda x: x.get("content", "") and x.get(
            "content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="TERMINATE",
        function_map={
            "search": search,
            "scrape": scrape,
        }
    )

    client.initiate_chat(researcher, message=query)

    # set the receiver to be researcher, and get a summary of the research report
    client.stop_reply_at_receive(researcher)
    client.send(
        "Give me the research report that just generated again, return ONLY the report & reference links", researcher)

    # return the last message the expert received
    return client.last_message()["content"]


# Define write content function
def write_content(research_material, topic):
    editor = autogen.AssistantAgent(
        name="editor",
        system_message="You are a senior editor of an AI blogger, you will define the structure of a short blog post based on material provided by the researcher, and give it to the writer to write the blog post",
        llm_config={"config_list": config_list},
    )

    writer = autogen.AssistantAgent(
        name="writer",
        system_message="You are a professional AI blogger who is writing a blog post about AI, you will write a short blog post based on the structured provided by the editor, and feedback from reviewer; After 2 rounds of content iteration, add TERMINATE to the end of the message",
        llm_config={"config_list": config_list},
    )

    reviewer = autogen.AssistantAgent(
        name="reviewer",
        system_message="You are a world class hash tech blog content critic, you will review & critic the written blog and provide feedback to writer.After 2 rounds of content iteration, add TERMINATE to the end of the message",
        llm_config={"config_list": config_list},
    )

    client = autogen.UserProxyAgent(
        name="admin",
        system_message="A human admin. Interact with editor to discuss the structure. Actual writing needs to be approved by this admin.",
        code_execution_config=False,
        is_termination_msg=lambda x: x.get("content", "") and x.get(
            "content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="TERMINATE",
    )

    groupchat = autogen.GroupChat(
        agents=[client, editor, writer, reviewer],
        messages=[],
        max_round=20)
    manager = autogen.GroupChatManager(groupchat=groupchat)

    client.initiate_chat(
        manager, message=f"Write a blog about {topic}, here are the material: {research_material}")

    client.stop_reply_at_receive(manager)
    client.send(
        "Give me the blog that just generated again, return ONLY the blog, and add TERMINATE in the end of the message", manager)

    # return the last message the expert received
    return client.last_message()["content"]


# Define content assistant agent
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

client = autogen.UserProxyAgent(
    name="Client",
    system_message="A human client. Interact with the planner to discuss the strategy. Plan execution needs to be approved by this client.",
    code_execution_config=False,
)
    
agency_strategist = autogen.AssistantAgent(
    name="Agency_Strategist",
    llm_config={"config_list": config_list},
    system_message=f'''
    You are the Lead Strategist.
    Your primary responsibility is to draft strategic briefs that effectively position our client's brand in the market.
    Based on the information provided in: {user_task}, your task is to craft a comprehensive strategic brief that outlines the brand's positioning, key messages, and strategic initiatives.
    The brief should delve deep into the brand's unique value proposition, target audience, and competitive landscape. 
    It should also provide clear directives on how the brand should be perceived and the emotions it should evoke.
    Once you've drafted the brief, it will be reviewed and iterated upon based on feedback from the client and our internal team. 
    Ensure that the brief is both insightful and actionable, setting a clear path for the brand's journey ahead.
    Collaborate with the Agency Researcher to ensure that the strategic brief is grounded in solid research and insights.
    '''
)

agency_researcher = autogen.AssistantAgent(
    name="Agency_Researcher",
    llm_config={"config_list": config_list},
    system_message=f'''
    You are the Lead Researcher. 
    Your primary responsibility is to delve deep into understanding user pain points, identifying market opportunities, and analyzing prevailing market conditions.
    Using the information from {user_task}, conduct thorough research to uncover insights that can guide our strategic decisions. 
    Your findings should shed light on user behaviors, preferences, and challenges.
    Use the research function to collect the latest information about a given topic, and then use the write_content function to write a very well-written content. 
    Additionally, assess the competitive landscape to identify potential gaps and opportunities for our client's brand. 
    Your research will be pivotal in shaping the brand's direction and ensuring it resonates with its target audience.
    Ensure that your insights are both comprehensive and actionable, providing a clear foundation for our subsequent strategic initiatives.
    Share your research findings with the Agency Strategist and Agency Marketer to inform the strategic and marketing initiatives.
    Reply TERMINATE when your task is done.
    ''',
    function_map={
        "research": research
    }
)

agency_designer = autogen.AssistantAgent(
    name="Agency_Designer",
    llm_config={"config_list": config_list},
    system_message=f'''
    You are the Lead Designer.
    Your primary responsibility is to transform strategic and marketing ideas into compelling visual narratives.
    Drawing from the direction given in {user_task}, craft designs, layouts, and visual assets that align with the brand's identity and resonate with its target audience.
    Work closely with the Creative Director and Agency Marketer to ensure that your designs align with the creative vision and marketing objectives.
    Your expertise will ensure that our client's brand is visually captivating and stands out in the market.
    '''
)

agency_writer = autogen.AssistantAgent(
    name="Agency_Copywriter",
    llm_config={"config_list": config_list},
    system_message=f'''
    You are the Lead Copywriter.
    Your primary role is to craft compelling narratives and messages that align with the brand's strategy and resonate with its audience.
    Based on the strategic direction from {user_task}, create engaging content, from catchy headlines to in-depth articles, ensuring that the brand's voice is consistent and impactful.
    Collaborate closely with the Agency Designer and Agency Marketer to ensure that text and visuals complement each other, creating a cohesive brand story.
    You can use the research function to collect the latest information about a given topic, and then use the write_content function to write a very well-written content. Reply TERMINATE when your task is done.
    ''',
    function_map={
        "write_content": write_content,
        "research": research,
    }
)

#writing_assistant = autogen.AssistantAgent(
#    name="writing_assistant",
#    system_message="You are a writing assistant, you can use research function to collect latest information about a given topic, and then use write_content function to write a very well written content; Reply TERMINATE when your task is done",
#    llm_config=llm_config_content_assistant,
#)

agency_marketer = autogen.AssistantAgent(
    name="Agency_Marketer",
    llm_config={"config_list": config_list},
    system_message=f'''
    You are the Lead Marketer. 
    Your primary role is to take the strategy and insights derived from research and transform them into compelling marketable ideas that resonate with the target audience.
    Using the strategic direction from {user_task}, craft innovative marketing campaigns, promotions, and initiatives that effectively communicate the brand's value proposition.
    Your expertise will bridge the gap between strategy and execution, ensuring that the brand's message is not only clear but also captivating. It's essential that your ideas are both impactful and aligned with the brand's overall vision.
    Collaborate with other teams to ensure a cohesive approach, and always strive to push the boundaries of creativity to set our client's brand apart in the market.
    Work in tandem with the Agency Manager to ensure that marketing initiatives align with the project's milestones and timelines.
    '''
)

agency_mediaplanner = autogen.AssistantAgent(
    name="Agency_Media_Planner",
    llm_config={"config_list": config_list},
    system_message=f'''
    You are the Lead Media Planner.
    Your main responsibility is to identify the best mix of media channels to deliver an advertising message to a clients' target audience.
    Using insights from {user_task}, strategize the most effective way to get a brand's message to its audience, whether it be through traditional media, digital platforms, or a combination of both.
    Collaborate closely with the Marketer and Manager to ensure that campaigns are executed effectively and within budget.
    '''
)

agency_manager = autogen.AssistantAgent(
    name="Agency_Manager",
    llm_config={"config_list": config_list},
    system_message=f'''
    You are the Project Manager. 
    Your primary responsibility is to oversee the entire project lifecycle, ensuring that all agents are effectively fulfilling their objectives and tasks on time.
    Based on the directives from {user_task}, coordinate with all involved agents, set clear milestones, and monitor progress. Ensure that user feedback is promptly incorporated, and any adjustments are made in real-time to align with the project's goals.
    Act as the central point of communication, facilitating collaboration between teams and ensuring that all deliverables are of the highest quality. Your expertise is crucial in ensuring that the project stays on track, meets deadlines, and achieves its objectives.
    Regularly review the project's status, address any challenges, and ensure that all stakeholders are kept informed of the project's progress.
    Coordinate with the Agency Director for periodic reviews and approvals, ensuring that the project aligns with the creative vision.
    '''
)
agency_director = autogen.AssistantAgent(
    name="Agency_Director",
    llm_config={"config_list": config_list},
    system_message=f'''
    You are the Creative Director at SCTY. Your primary role is to guide the creative vision of the project, ensuring that all ideas are not only unique and compelling but also meet the highest standards of excellence and desirability.
    Drawing from the insights of {user_task}, oversee the creative process, inspire innovation, and set the bar for what's possible. Challenge the team to think outside the box and push the boundaries of creativity.
    Review all creative outputs, provide constructive feedback, and ensure that every piece aligns with the brand's identity and resonates with the target audience. 
    Your expertise is pivotal in ensuring that our work stands out in the market and leaves a lasting impact.
    Collaborate closely with all teams, fostering a culture of excellence, and ensuring that our creative solutions are both groundbreaking and aligned with the project's objectives.
    Engage with the Agency Strategist and Agency Marketer to ensure that the creative outputs align with the strategic direction and marketable ideas.
    '''
)

agency_accountmanager = autogen.AssistantAgent(
    name="Agency_Account_Manager",
    llm_config={"config_list": config_list},
    system_message=f'''
    You are the Account Manager.
    Your primary responsibility is to nurture the relationship between the agency and the client, ensuring clear communication and understanding of the client's needs and feedback.
    Act as a bridge between the client and the agency, facilitating collaboration, and ensuring that the project aligns with the client's expectations.
    Regularly update the client on the project's status and ensure that their feedback is incorporated effectively.
    Collaborate with the Agency Manager to ensure that timelines are met and deliverables are of the highest quality.
    '''
)

groupchat = autogen.GroupChat(agents=[
    client, agency_researcher, agency_strategist, agency_writer, agency_designer, agency_mediaplanner, agency_marketer, agency_manager, agency_director, agency_accountmanager], messages=[], max_round=20)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})

client.initiate_chat(
    manager,
    message=f"""
    {user_task}
    """,
)

client = autogen.UserProxyAgent(
    name="client",
    human_input_mode="TERMINATE",
    function_map={
        "write_content": write_content,
        "research": research,
    }
)

with open('logs/output.md', 'w') as f:
    f.write(str(autogen.ChatCompletion.logged_history))
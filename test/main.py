import os
import requests
from bs4 import BeautifulSoup
import json
import autogen
import openai
from autogen import OpenAIWrapper, config_list_from_json
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate 
from dotenv import load_dotenv

load_dotenv()
config_list = autogen.config_list_from_dotenv(dotenv_file_path='.env')
openai.api_key = os.getenv("OPENAI_API_KEY")
BROWSERLESS_API_KEY = os.getenv("BROWSERLESS_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

brand_task = input("Please enter the brand or company name: ")
user_task = input("Please enter the your goal, brief, or problem statement: ")

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

    researcher = autogen.AssistantAgent(
        name="researcher",
        system_message="Research about a given query, collect as many information as possible, and generate detailed research results with loads of technique details with all reference links attached; Add TERMINATE to the end of the research report;",
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

    user_proxy.initiate_chat(researcher, message=query)
    user_proxy.stop_reply_at_receive(researcher)
    user_proxy.send(
        "Give me the research report that just generated again, return ONLY the report & reference links", researcher)

    # return the last message the expert received
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

    # return the last message the expert received
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

agency_strategist = autogen.AssistantAgent(
    name="Agency_Strategist",
    llm_config={"config_list": config_list},
    system_message=f'''
    You are the Lead Strategist.
    Your primary responsibility is to draft strategic briefs that effectively position our client's brand in the market.
    Based on the information provided for {brand_task} on {user_task}, your task is to craft a comprehensive strategic brief that outlines the brand's positioning, key messages, and strategic initiatives.
    The brief should delve deep into the brand's unique value proposition, target audience, and competitive landscape. 
    It should also provide clear directives on how the brand should be perceived and the emotions it should evoke.
    Once you've drafted the brief, it will be reviewed and iterated upon based on feedback from the client and our internal team. 
    Ensure that the brief is both insightful and actionable, setting a clear path for the brand's journey ahead.
    Collaborate with the Agency Researcher to ensure that the strategic brief is grounded in solid research and insights.
    Be concise and not verbose. Refrain from any conversations that don't serve the goal of the user.
    ''',
    function_map={
        "research": research,
    }
)

agency_researcher = autogen.AssistantAgent(
    name="Agency_Researcher",
    llm_config={"config_list": config_list},
    system_message=f'''
    You are the Lead Researcher. 
    You must use the research function to provide a topic for the writing_assistant in order to get up to date information outside of your knowledge cutoff
    Your primary responsibility is to delve deep into understanding user pain points, identifying market opportunities, and analyzing prevailing market conditions.
    Using the information from {user_task}, conduct thorough research to uncover insights that can guide our strategic decisions. 
    Your findings should shed light on user behaviors, preferences, and challenges.
    Additionally, assess the competitive landscape to identify potential gaps and opportunities for our client's brand. 
    Your research will be pivotal in shaping the brand's direction and ensuring it resonates with its target audience.
    Ensure that your insights are both comprehensive and actionable, providing a clear foundation for our subsequent strategic initiatives.
    Share your research findings with the Agency Strategist and Agency Marketer to inform the strategic and marketing initiatives.
    Be concise and not verbose. Refrain from any conversations that don't serve the goal of the user.
    ''',
    function_map={
        "research": research,
    }
)

agency_writer = autogen.AssistantAgent(
    name="Agency_Copywriter",
    llm_config={"config_list": config_list},
    system_message=f'''
    You are the Lead Copywriter.
    Your primary role is to craft compelling narratives and messages that align with the brand's strategy and resonate with its audience.
    Based on the strategic direction from {user_task}, create engaging content, from catchy headlines to in-depth articles, ensuring that the brand's voice is consistent and impactful.
    Collaborate closely with the Agency Designer and Agency Marketer to ensure that text and visuals complement each other, creating a cohesive brand story.
    Be concise and not verbose. Refrain from any conversations that don't serve the goal of the user.
    ''',
    function_map={
        "write_content": write_content,
    }
)

writing_assistant = autogen.AssistantAgent(
    name="writing_assistant",
    llm_config=llm_config_content_assistant,
    system_message=f'''You are a writing assistant, you can use research function to collect latest information about a given topic, 
    and then use write_content function to write a very well written content;
    Reply TERMINATE when your task is done
    Be concise and not verbose. Refrain from any conversations that don't serve the goal of the user.
    ''',
    function_map={
        "research": research,
    }
)

agency_marketer = autogen.AssistantAgent(
    name="Agency_Marketer",
    llm_config={"config_list": config_list},
    system_message=f'''
    You are the Lead Marketer. Be concise and avoid pleasantries. Refrain from any conversations that don't serve the goal of the user, ie. thank you.
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
    You are the Lead Media Planner. Be concise and avoid pleasantries. Refrain from any conversations that don't serve the goal of the user, ie. thank you.
    Your main responsibility is to identify the best mix of media channels to deliver an advertising message to a clients' target audience.
    Using insights from {user_task}, strategize the most effective way to get a brand's message to its audience, whether it be through traditional media, digital platforms, or a combination of both.
    Collaborate closely with the Marketer and Manager to ensure that campaigns are executed effectively and within budget.
    '''
)

agency_manager = autogen.AssistantAgent(
    name="Agency_Manager",
    llm_config={"config_list": config_list},
    system_message=f'''
    You are the Project Manager. Be concise and avoid pleasantries. Refrain from any conversations that don't serve the goal of the user, ie. thank you.
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
    You are the Creative Director at SCTY. Be concise and avoid pleasantries. Refrain from any conversations that don't serve the goal of the user, ie. thank you.
    Your primary role is to guide the creative vision of the project, ensuring that all ideas are not only unique and compelling but also meet the highest standards of excellence and desirability.
    Drawing from the insights of {user_task}, oversee the creative process, inspire innovation, and set the bar for what's possible. Challenge the team to think outside the box and push the boundaries of creativity.
    Review all creative outputs, provide constructive feedback, and ensure that every piece aligns with the brand's identity and resonates with the target audience. 
    Your expertise is pivotal in ensuring that our work stands out in the market and leaves a lasting impact.
    Collaborate closely with all teams, fostering a culture of excellence, and ensuring that our creative solutions are both groundbreaking and aligned with the project's objectives.
    Engage with the Agency Strategist and Agency Marketer to ensure that the creative outputs align with the strategic direction and marketable ideas.
    '''
)

""" agency_accountmanager = autogen.AssistantAgent(
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
) """

user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    is_termination_msg=lambda x: x.get("content", "").strip().upper() == "TERMINATE",
    human_input_mode="TERMINATE",
    system_message="Provide the necessary research input. Save all output to markdown file",
    code_execution_config={},
    function_map={
        "research": research,
        "write_content": write_content,
    }
)

user_proxy.register_function(
    function_map={
        "research": research,
        "write_content": write_content,
    }
)

groupchat = autogen.GroupChat(agents=[
    user_proxy, agency_manager, agency_researcher, agency_strategist, agency_writer, writing_assistant, agency_marketer, agency_mediaplanner, agency_director], messages=[], max_round=20)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})

user_proxy.initiate_chat(
    manager, 
    message=user_task,
)
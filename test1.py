from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
import os
from datetime import datetime
from dotenv import load_dotenv
from cachetools import cached, TTLCache
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SerpAPIWrapper as BaseSerpAPIWrapper

# Load environment variables from .env file
load_dotenv()

config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")

# Adjust the cache size (maxsize) and expiration time (ttl) as needed
cache = TTLCache(maxsize=100, ttl=3600)

class SerpAPIWrapper(BaseSerpAPIWrapper):
    @cached(cache)
    def results(self, query: str) -> dict:
        """Run query through SerpAPI and return the raw result (with caching)."""
        return super().results(query)

def web_search(query: str) -> str:
    """Search the web using the provided query and return the results."""
    serpapi = SerpAPIWrapper()
    results = serpapi.results(query)
    # Extract relevant info from results. Depending on the SerpAPI result format, you may need to adjust this.
    search_results = results.get('organic_results', [])
    # Convert the results into a string
    results_str = "\n".join([result.get('title', '') for result in search_results])
    return results_str

# Define the AssistantAgent
class QueryingAssistant(autogen.AssistantAgent):
    def reply(self, message_content: str) -> str:
        # For demonstration, the assistant simply sends the received message content back.
        # This could be adjusted to refine the query or decide when to invoke the web search.
        return message_content

# Configuration
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")

# Create AssistantAgent instance
chatbot = QueryingAssistant(
    name="chatbot",
    llm_config={"config_list": config_list},
    system_message="For coding tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
)

# Create an instance of the UserProxyAgent
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "coding"},
    llm_config={"config_list": config_list},
)

# Register the web search function
user_proxy.register_function(
    function_map={
        "web_search": web_search,
    }
)

# Initiate the conversation with chatbot as the initiator and user_proxy as the recipient
chatbot.initiate_chat(
    user_proxy,
    message=f"Search the web for: {user_task}",
)

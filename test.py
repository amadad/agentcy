import autogen
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from cachetools import cached, TTLCache
from langchain.utilities import SerpAPIWrapper as BaseSerpAPIWrapper


user_task = input("Please enter the task for the new client's campaign brief: ")

config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")

# Adjust the cache size (maxsize) and expiration time (ttl) as needed
cache = TTLCache(maxsize=100, ttl=3600)

class SerpAPIWrapper(BaseSerpAPIWrapper):
    @cached(cache)
    def results(self, query: str) -> dict:
        """Run query through SerpAPI and return the raw result (with caching)."""
        return super().results(query)

class QueryingAssistant(autogen.AssistantAgent):
    def reply(self, message_content: str) -> str:
        """Override the reply method to send a query."""
        # For demonstration, the assistant simply sends the received message content back.
        # In a real-world scenario, you could add logic to refine the query or decide when to query.
        return message_content

# Define a UserProxyAgent with web search capabilities
class WebSearchUserProxy(autogen.UserProxyAgent):
    
    def search_web(self, query: str) -> list:
        """Search the web using the provided query and return the results."""
        print("Attempting to search the web...")  # Debug print
        try:
            serpapi = SerpAPIWrapper()
            results = serpapi.results(query)
            print("Search results obtained.")  # Debug print
            return results.get('organic_results', [])
        except Exception as e:
            print(f"Error during search: {e}")  # Print error for debugging
            return []
    
    def reply(self, message_content: str) -> str:
        """Override the reply method to perform a web search."""
        print("Reply method called.")  # Debug print
        search_results = self.search_web(message_content)
        results_str = "\n".join([result.get('title', '') for result in search_results])
        return f"Here are the search results for your query:\n{results_str}"

# Configuration
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")

# Create AssistantAgent instance
chatbot = QueryingAssistant(
    name="chatbot",
    llm_config={"config_list": config_list},
    system_message="For coding tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
)

# Create an instance of the custom UserProxyAgent
user_proxy = WebSearchUserProxy(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "coding"},
    llm_config={"config_list": config_list},
)

# Initiate the conversation with chatbot as the initiator and user_proxy as the recipient
chatbot.initiate_chat(
    user_proxy,
    message=f"""
    {user_task}
    """,
)
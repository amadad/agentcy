# Agentcy: Multi-Agent Creative Collaboration

<p align="center">
  <img src='./misc/logo1.png' width=888>
</p>

A small team autonomous of agents help you unlock, uncover or explore the potential of your business. Agentcy takes two simple inputs to generate a plan, research and deliverables to help you gain a competitive advantahe.

## 📖 Overview

Modeled on advertising and creative agencies, you'll move from problem space to solution space with successive steps of research, writing, ideation and strategic planning. Agentcy uses the [AutoGen framework](https://github.com/microsoft/autogen) to orchestrate multiple agents to ensure that tasks are handled by the most qualified agent, leading to more efficient and accurate outcomes.

## 🕵🏽 Agents

The agents involved in the collaboration include:

1. **Agency Manager**: Creates a plan and approach to tackle the problem or opportunity.
2. **Agency Researcher**: Conducts research on user pain points, market opportunities, and prevailing market conditions.
2. **Writing Assistant**: Utilizes research and content writing functions to generate content.
3. **Agency Strategist**: Drafts strategic briefs for effective brand positioning in the market.
4. **Agency Copywriter**: Crafts compelling narratives and messages that align with the brand's strategy.
5. **Agency Media Planner**: Identifies the best mix of media channels for advertising.
6. **Agency Marketer**: Transforms strategy and insights into marketable ideas.
8. **Agency Director**: Guides the creative vision of the project.
9. **User Proxy**: Acts as an intermediary between the human user and the agents.

## 🛠️ Tools Used

1. `Serper` for realtime web search
2. `Browserless` for web scrape
3. `Langchain` for content summarization 

<p align="center">
  <img src='./misc/flow.png' width=888>
</p>

## ⚙️ Setup & Configuration

1. Ensure required libraries are installed:
```
pip install pyautogen
```

2. Set up the OpenAI configuration list by either providing an environment variable `OAI_CONFIG_LIST` or specifying a file path.
```
[
    {
        "model": "gpt-3.5-turbo", #or whatever model you prefer
        "api_key": "INSERT_HERE"
    }
]
```

3. Setup api keys in .env:
```
OPENAI_API_KEY="XXX"
SERPAPI_API_KEY="XXX"
SERPER_API_KEY="XXX"
BROWSERLESS_API_KEY="XXX"
```

4. Launch in CLI:
```
python3 main.py
```

## ⏯️ Conclusion

In the realm of creative agencies, the multi-agent collaboration approach revolutionizes the way projects are handled. By tapping into the distinct expertise of various agency roles, from strategists to media planners, we can guarantee that each facet of a project is managed by those best suited for the task. This methodology not only ensures precision and efficiency but also showcases its versatility, as it can be tailored to suit diverse project requirements, whether it's brand positioning, content creation, or any other creative endeavor. 

Credit to [Jason Zhou's RAG example](https://github.com/JayZeeDesign/microsoft-autogen-experiments).

## 📈 Roadmap

[x] (Refine workflow and data pass through to agents)
[x] (Reduce unnecessary back and forth)
[ ] (Implement alternative and local LLM models)
[ ] (Save files to local folder)
[ ] (Implement other agents, see commented out agents)
[ ] (Create and train fine-tuned agents for each domain specific task)
[ ] (Add more tools for agents to utilize)
[ ] (Create UI for project)

## 📝 License 

MIT License. See [LICENSE](https://opensource.org/license/mit/) for more information.

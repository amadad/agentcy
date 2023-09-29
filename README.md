# Multi-Agent Collaboration with AutoGen

<p align="center">
  <img src='./misc/logo1.png' width=600>
</p>

This code demonstrates the power of multi-agent collaboration using the AutoGen library. Instead of relying on a single agent to handle tasks, multiple specialized agents work together, each bringing its expertise to the table.

## 📖 Overview

The code sets up a collaborative environment where multiple agents, each with its unique role and expertise, come together to discuss, plan, and execute tasks. This collaboration ensures that different aspects of a task are handled by the most qualified agent, leading to more efficient and accurate outcomes.

## 🕵🏽 Agents

Here are the agents involved in the collaboration:

1. **Client**: Represents the human user. Interacts with the planner to discuss and approve the plan.
2. **Strategist**: Drafts strategic briefs that effectively position the client's brand in the market based on comprehensive research and insights.
3. **Researcher**: Delves deep into understanding user pain points, identifies market opportunities, and analyzes prevailing market conditions.
4. **Marketer**: Transforms strategy and insights into compelling marketable ideas that resonate with the target audience.
5. **Manager**: Oversees the entire project lifecycle, ensuring all agents are effectively fulfilling their objectives and tasks on time.
6. **Director**: Guides the creative vision of the project, ensuring ideas are unique, compelling, and meet the highest standards of excellence.


## 🤝 Collaboration Flow

1. The `GroupChat` class is used to create a collaborative environment where all agents can communicate.
2. The `GroupChatManager` manages the group chat, ensuring smooth communication between agents.
3. The `initiate_chat` method is called to start the collaboration. In this example, the task is to plan a month-long trip to Bangkok.

## ⚙️ Setup & Configuration

1. Ensure you have the required libraries installed:

```
pip install openai autogen
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

3. Instantiate each agent with its unique name, configuration, and system message.

4. Create a group chat with all the agents and initiate the collaboration.

## ⏯️ Conclusion

This multi-agent collaboration approach allows for more comprehensive and efficient task handling. By leveraging the expertise of multiple agents, we can ensure that every aspect of a task is addressed by the most qualified entity. Whether it's planning a trip, as demonstrated in this example, or any other task, this collaborative approach can be adapted to fit various scenarios.
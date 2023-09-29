# Multi-Agent Collaboration with AutoGen

This code demonstrates the power of multi-agent collaboration using the AutoGen library. Instead of relying on a single agent to handle tasks, multiple specialized agents work together, each bringing its expertise to the table.

## Overview

The code sets up a collaborative environment where multiple agents, each with its unique role and expertise, come together to discuss, plan, and execute tasks. This collaboration ensures that different aspects of a task are handled by the most qualified agent, leading to more efficient and accurate outcomes.

## Agents

Here are the agents involved in the collaboration:

1. **Admin (User Proxy)**: Represents the human user. Interacts with the planner to discuss and approve the plan.
2. **Fun Manager**: Focuses on maximizing fun and optimizing for unique memorable experiences.
3. **Gym Trainer**: Ensures the user maintains a healthy lifestyle by recommending training and diet plans.
4. **Executive Assistant**: Prioritizes daily work tasks, ensuring they are completed before any recreational activities.
5. **Planner**: Suggests and revises plans based on feedback from other agents and the admin.
6. **Critic**: Reviews the plan to ensure all objectives are met and provides feedback.

## Collaboration Flow

1. The `GroupChat` class is used to create a collaborative environment where all agents can communicate.
2. The `GroupChatManager` manages the group chat, ensuring smooth communication between agents.
3. The `initiate_chat` method is called to start the collaboration. In this example, the task is to plan a month-long trip to Bangkok.

## Setup & Configuration

1. Ensure you have the required libraries installed:
pip install openai autogen


2. Set up the OpenAI configuration list by either providing an environment variable `OAI_CONFIG_LIST` or specifying a file path.

3. Instantiate each agent with its unique name, configuration, and system message.

4. Create a group chat with all the agents and initiate the collaboration.

## Conclusion

This multi-agent collaboration approach allows for more comprehensive and efficient task handling. By leveraging the expertise of multiple agents, we can ensure that every aspect of a task is addressed by the most qualified entity. Whether it's planning a trip, as demonstrated in this example, or any other task, this collaborative approach can be adapted to fit various scenarios.
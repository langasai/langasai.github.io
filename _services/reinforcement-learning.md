---
title: "Reinforcement Learning"
date: 2019-03-28T15:14:54+10:00
weight: 5
---

Reinforcement learning is a powerful paradigm that draws parallels to how humans learn. While a largely experimental approach, there is significant theory and empirical evidence that supports the approach.

From robotics to recommendation systems, reinforcement learning is a powerful tool for solving complex business and engineering problems.

---

## TL;DR - Key Business Takeaways

- **Evolving environments:** Reinforcement learning is a powerful tool that allows online adaptation to handle an ever-evolving environment.
- **Human alignment:** Human preferences are often difficult to define, but reinforcement learning abstracts this problem away to certain extents.
- **Robotics Applications:** In a world where automation is becoming more prevalent, reinforcement learning is often more appropriate than direct deep learning approaches.
- **Business Planning Questions (not limited to):** Do I have an ever evolving environment to model? Do I have an expert/customer in the loop to provide feedback? Do I want a solution that mimics human preferences?

# What is Reinforcement Learning?

Reinforcement learning is a type of machine learning that involves an agent learning to make decisions by interacting with an environment. The agent learns to maximize a reward signal by taking actions in the environment. 

Example problems that reinforcement learning can be useful in:
- Robotics: We want a robot to learn how to move an item from any given start position to any given end position.
- Recommendation Systems: We want a system to learn how to recommend items to a user based on their ever evolving preferences.
- Game Playing: We want a system to learn how to play a game like chess - being able to handle any type of opponent move.
- Financial Trading Systems: Given a history of asset prices and other meta information, we want to determine our next position in the market.
- Language Modeling: We want a system to generate text that a human prefers over a baseline text generation. 

# How does Reinforcement Learning work?

Technically speaking, reinforcement learning aims to teach an agent how to take actions in an environment to maximize a reward signal.

The environment is defined as a Markov Decision Process (MDP), which is a tuple of:
- States: The set of possible states the agent can be in.
- Actions: The set of possible actions the agent can take.
- Transitions: The probability of transitioning from one state to another given an action.
- Rewards: The reward the agent receives for each state-action pair.

In the example of a recommendation system for movies, the environment could be the set of all movies, and the agent is the recommendation system. The agent's possible actions are movies to recommend, and the user's review score of the movie could be the reward. The states are thus all possible sequences of movies the user has seen up to the current time point. In an ideal world, we would know how likely a user is to select a movie, conditioned on the previous movies they have seen, and the probability they would like the movie, thereby giving us the probability of transitioning from one state to another given an action. However, this is typically unknown, and we must learn it through experience. This is the core idea of reinforcement learning.

# How do we solve Reinforcement Learning problems?

There are two main approaches to solving reinforcement learning problems:
- Model-based: We attempt to model the environment and use this model as the transition function.
- Model-free: We learn the optimal policy directly from experience without modeling the environment.

A model-based approach may learn faster, but if the transition function is not accurate, the agent may not learn the optimal policy. A model-free approach may learn slower, but does not require any knowledge of the environment. 

Two additional approaches are:

- Policy gradient methods: We learn the optimal policy directly by optimizing for the reward signal.
- Value/Q-learning: We learn the optimal policy by learning the optimal action-value function.

In practice, most state-of-the-art reinforcement learning algorithms are model-free, and use a combination of policy gradient methods and Q-learning. 

In fact, signal from both the the policy gradient algorithm typically learns an action through deep learning methods, but also takes in signal coming from the Q-learning algorithm, which is also learned through deep learning methods. 

While reinforcement learning is a powerful tool, however it requires a lot of experience and knowledge to implement properly. 

# Where has reinforcement learning been successful?

In our consulting practice, we have seen successful use cases of reinforcement learning in the following areas: 

- Language Modelling in reasoning: OpenAI's GPT-o1 was trained using reinforcement learning. Deepseek R1 was also trained using reinforcement learning. Ultimately, our most successful clients in reinforcement learning have been in the language modelling space. 
- Recommendation Systems: Reinforcement learning has been used to create recommendation systems that can adapt to user preferences and changing contexts. Many clients are using reinforcement learning in their customer analytics pipelines, studying things like click-through rate and improving their SEO methods.
- Robotics: Current state-of-the-art robotic manipulation tasks are solved using reinforcement learning.
- Server management: Reinforcement learning has been used to create systems that can optimize server resources and reduce costs, as a function of user load, ambient temperature, energy consumption, etc.
- Game Playing: Reinforcement learning has been used to create agents that can play games like chess and Go at a superhuman level. Atari games have been played at a superhuman level (scores higher than human players).

# What does this mean for your business?

While all these methods are still not perfect, the reinforcement learning paradigm has brought a lot of improvements not otherwise achievable with conventional deep learning methods. We find that our most successful clients working on reinforcement learning problems are those that have a problem where there is a clear way to formulate the problem as an environment, with states, actions and rewards. We can help you define the problem, and then implement a solution using reinforcement learning. Book a time for a consult with us and we can begin our roadmap for how LLMs can better serve your business. 








import gym
import logging
from tella.agent import Agent
from logging_agent import LoggingAgent

"""
Example usage of an Agent.

In this case we assume there is only 1 block with 1 task and 1 episode for simplicity.
"""


def main():
    logging.basicConfig(level=logging.INFO)

    env: gym.Env = gym.make("CartPole-v1")
    agent: Agent = LoggingAgent(env.observation_space, env.action_space)

    agent.block_start(is_learning_allowed=True)

    agent.task_start(task_name="CartPole", variant_name="Default")

    agent.episode_start()

    obs, done = env.reset(), False
    while not done:
        action = agent.step_observe(obs)
        next_obs, reward, done, info = env.step(action)
        agent.step_reward(obs, action, reward, done, next_obs)

    agent.episode_end()

    agent.task_end(task_name="CartPole", variant_name="Default")

    agent.block_end(is_learning_allowed=True)


if __name__ == "__main__":
    main()

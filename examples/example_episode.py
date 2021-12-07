import gym
import logging
from tella.agents.continual_rl_agent import ContinualRLAgent
from rl_logging_agent import LoggingAgent

"""
Example usage of an ContinualRLAgent.

In this case we assume there is only 1 block with 1 task and 1 episode for simplicity.
"""


def main():
    logging.basicConfig(level=logging.INFO)

    env: gym.Env = gym.make("CartPole-v1")
    agent: ContinualRLAgent = LoggingAgent(
        env.observation_space, env.action_space, num_envs=1
    )

    agent.block_start(is_learning_allowed=True)

    agent.task_start(task_name="CartPole", variant_name="Default")

    obs, done = env.reset(), False
    while not done:
        action = list(agent.step_observe([obs]))[0]
        next_obs, reward, done, info = env.step(action)
        agent.step_transition((obs, action, reward, done, next_obs))

    agent.task_end(task_name="CartPole", variant_name="Default")

    agent.block_end(is_learning_allowed=True)


if __name__ == "__main__":
    main()
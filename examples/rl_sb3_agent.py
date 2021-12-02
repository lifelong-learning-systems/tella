import typing
import gym
from tella.curriculum import Curriculum, Block, EvalBlock, LearnBlock
from tella.experiences.rl import (
    LimitedEpisodesExperience,
    RLExperience,
)
from tella.run import run
##TODO add to tella instead of examples
from sb3_agent import ContinualRLSB3Agent
from stable_baselines3 import PPO


class ExampleCurriculum(Curriculum[RLExperience]):
    def blocks(self) -> typing.Iterable[Block]:
        yield LearnBlock(
            [LimitedEpisodesExperience(lambda: gym.make("CartPole-v1"), num_episodes=5)]
        )
        yield EvalBlock(
            [LimitedEpisodesExperience(lambda: gym.make("CartPole-v1"), num_episodes=5)]
        )


if __name__ == "__main__":
    import logging
    from rl_logging_agent import LoggingAgent

    logging.basicConfig(level=logging.INFO)

    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=1)
    agent = ContinualRLSB3Agent(model,1)
    curriculum = ExampleCurriculum()

    run(agent, curriculum)
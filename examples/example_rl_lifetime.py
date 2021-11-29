import typing
import gym
from tella.curriculum import Curriculum, Block, EvalBlock, LearnBlock
from tella.experiences.rl import (
    LimitedEpisodesExperience,
    RLExperience,
)
from tella.run import run


class ExampleCurriculum(Curriculum[RLExperience]):
    def blocks(self) -> typing.Iterable[Block]:
        yield LearnBlock(
            [LimitedEpisodesExperience(lambda: gym.make("CartPole-v1"), num_episodes=1)]
        )
        yield EvalBlock(
            [LimitedEpisodesExperience(lambda: gym.make("CartPole-v1"), num_episodes=1)]
        )


if __name__ == "__main__":
    import logging
    from rl_logging_agent import LoggingAgent

    logging.basicConfig(level=logging.INFO)

    env = gym.make("CartPole-v1")
    agent = LoggingAgent(env.observation_space, env.action_space, num_envs=1)
    curriculum = ExampleCurriculum()

    run(agent, curriculum)

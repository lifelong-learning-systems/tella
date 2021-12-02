import typing
import gym
from tella.curriculum import AbstractCurriculum, AbstractLearnBlock, AbstractEvalBlock
from tella.curriculum.rl_task_variant import AbstractRLTaskVariant, EpisodicTaskVariant
from tella.curriculum.builders import simple_learn_block, simple_eval_block
from tella.run import run


class ExampleCurriculum(AbstractCurriculum[AbstractRLTaskVariant]):
    def learn_blocks_and_eval_blocks(
        self,
    ) -> typing.Iterable[
        typing.Union[
            "AbstractLearnBlock[AbstractRLTaskVariant]",
            "AbstractEvalBlock[AbstractRLTaskVariant]",
        ]
    ]:
        yield simple_learn_block(
            [EpisodicTaskVariant(lambda: gym.make("CartPole-v1"), num_episodes=1)]
        )
        yield simple_eval_block(
            [EpisodicTaskVariant(lambda: gym.make("CartPole-v1"), num_episodes=1)]
        )


if __name__ == "__main__":
    import logging
    from rl_logging_agent import LoggingAgent

    logging.basicConfig(level=logging.INFO)

    env = gym.make("CartPole-v1")
    agent = LoggingAgent(env.observation_space, env.action_space, num_envs=1)
    curriculum = ExampleCurriculum()

    run(agent, curriculum)

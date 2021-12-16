import typing
import gym
from tella.curriculum import AbstractCurriculum, AbstractLearnBlock, AbstractEvalBlock
from tella.curriculum import AbstractRLTaskVariant, EpisodicTaskVariant
from tella.curriculum import simple_learn_block, simple_eval_block


class SimpleRLCurriculum(AbstractCurriculum[AbstractRLTaskVariant]):
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

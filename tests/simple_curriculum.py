import typing
from gym.envs.classic_control import CartPoleEnv
from tella.curriculum import AbstractCurriculum, AbstractLearnBlock, AbstractEvalBlock
from tella.curriculum import AbstractRLTaskVariant, EpisodicTaskVariant
from tella.curriculum import simple_learn_block, simple_eval_block


class SimpleRLCurriculum(AbstractCurriculum[AbstractRLTaskVariant]):
    def learn_blocks_and_eval_blocks(
        self,
        rng_seed: int,
    ) -> typing.Iterable[
        typing.Union[
            "AbstractLearnBlock[AbstractRLTaskVariant]",
            "AbstractEvalBlock[AbstractRLTaskVariant]",
        ]
    ]:
        yield simple_learn_block(
            [
                EpisodicTaskVariant(CartPoleEnv, num_episodes=1),
                EpisodicTaskVariant(
                    CartPoleEnv, num_episodes=1, variant_label="Variant1"
                ),
            ]
        )
        yield simple_eval_block([EpisodicTaskVariant(CartPoleEnv, num_episodes=1)])

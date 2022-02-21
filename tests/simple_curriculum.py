import typing
from gym.envs.classic_control import CartPoleEnv
from tella.curriculum import (
    AbstractCurriculum,
    Block,
    TaskVariant,
    simple_learn_block,
    simple_eval_block,
)


class SimpleRLCurriculum(AbstractCurriculum):
    def learn_blocks_and_eval_blocks(self) -> typing.Iterable[Block]:
        yield simple_learn_block(
            [
                TaskVariant(
                    CartPoleEnv,
                    num_episodes=1,
                    rng_seed=self.rng.bit_generator.random_raw(),
                ),
                TaskVariant(
                    CartPoleEnv,
                    num_episodes=1,
                    variant_label="Variant1",
                    rng_seed=self.rng.bit_generator.random_raw(),
                ),
            ]
        )
        yield simple_eval_block(
            [
                TaskVariant(
                    CartPoleEnv,
                    num_episodes=1,
                    rng_seed=self.rng.bit_generator.random_raw(),
                )
            ]
        )


class MultiEpisodeRLCurriculum(AbstractCurriculum):
    def learn_blocks_and_eval_blocks(self) -> typing.Iterable[Block]:
        yield simple_learn_block(
            [
                TaskVariant(
                    CartPoleEnv,
                    num_episodes=5,
                    rng_seed=self.rng.bit_generator.random_raw(),
                ),
                TaskVariant(
                    CartPoleEnv,
                    num_episodes=4,
                    variant_label="Variant1",
                    rng_seed=self.rng.bit_generator.random_raw(),
                ),
            ]
        )
        yield simple_eval_block(
            [
                TaskVariant(
                    CartPoleEnv,
                    num_episodes=3,
                    rng_seed=self.rng.bit_generator.random_raw(),
                )
            ]
        )


class LearnOnlyCurriculum(AbstractCurriculum):
    def learn_blocks_and_eval_blocks(self) -> typing.Iterable[Block]:
        yield simple_learn_block(
            [
                TaskVariant(
                    CartPoleEnv,
                    num_episodes=1,
                    rng_seed=self.rng.bit_generator.random_raw(),
                ),
                TaskVariant(
                    CartPoleEnv,
                    num_episodes=1,
                    variant_label="Variant1",
                    rng_seed=self.rng.bit_generator.random_raw(),
                ),
            ]
        )


class EvalOnlyCurriculum(AbstractCurriculum):
    def learn_blocks_and_eval_blocks(self) -> typing.Iterable[Block]:
        yield simple_eval_block(
            [
                TaskVariant(
                    CartPoleEnv,
                    num_episodes=1,
                    rng_seed=self.rng.bit_generator.random_raw(),
                ),
                TaskVariant(
                    CartPoleEnv,
                    num_episodes=1,
                    variant_label="Variant1",
                    rng_seed=self.rng.bit_generator.random_raw(),
                ),
            ]
        )

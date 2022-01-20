import typing
import tella
import gym
from tella.curriculum import AbstractTaskVariant, Transition
from tella.agents import ContinualRLAgent, Observation, Action


class SimpleRLAgent(ContinualRLAgent):
    def __init__(
        self,
        rng_seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        num_envs: int,
        config_file: typing.Optional[str] = None,
    ) -> None:
        super().__init__(
            rng_seed, observation_space, action_space, num_envs, config_file
        )
        self.all_events = []

    def block_start(self, is_learning_allowed: bool) -> None:
        self.all_events.append(
            (self.block_start, "learn" if is_learning_allowed else "eval")
        )
        return super().block_start(is_learning_allowed)

    def block_end(self, is_learning_allowed: bool) -> None:
        self.all_events.append(
            (self.block_end, "learn" if is_learning_allowed else "eval")
        )
        return super().block_end(is_learning_allowed)

    def task_start(self, task_name: typing.Optional[str]) -> None:
        self.all_events.append((self.task_start, task_name))
        return super().task_start(task_name)

    def task_end(self, task_name: typing.Optional[str]) -> None:
        self.all_events.append((self.task_end, task_name))
        return super().task_end(task_name)

    def task_variant_start(
        self, task_name: typing.Optional[str], variant_name: typing.Optional[str]
    ) -> None:
        self.all_events.append((self.task_variant_start, task_name, variant_name))
        return super().task_variant_start(task_name, variant_name)

    def task_variant_end(
        self, task_name: typing.Optional[str], variant_name: typing.Optional[str]
    ) -> None:
        self.all_events.append((self.task_variant_end, task_name, variant_name))
        return super().task_variant_end(task_name, variant_name)

    def learn_task_variant(self, task_variant: AbstractTaskVariant):
        self.all_events.append(
            (
                self.learn_task_variant,
                task_variant.task_label,
                task_variant.variant_label,
            )
        )
        return super().learn_task_variant(task_variant)

    def eval_task_variant(self, task_variant: AbstractTaskVariant):
        self.all_events.append(
            (
                self.eval_task_variant,
                task_variant.task_label,
                task_variant.variant_label,
            )
        )
        return super().eval_task_variant(task_variant)

    def choose_actions(
        self, observations: typing.List[typing.Optional[Observation]]
    ) -> typing.List[typing.Optional[Action]]:
        return [
            None if obs is None else self.action_space.sample() for obs in observations
        ]

    def receive_transitions(
        self, transitions: typing.List[typing.Optional[Transition]]
    ):
        pass

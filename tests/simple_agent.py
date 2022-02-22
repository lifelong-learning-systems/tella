"""
Copyright © 2021-2022 The Johns Hopkins University Applied Physics Laboratory LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import typing
import gym
from tella.curriculum import Transition
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

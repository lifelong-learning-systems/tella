"""
Copyright © 2021 The Johns Hopkins University Applied Physics Laboratory LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import abc
import typing
from ...curriculum.rl_task_variant import Transition


class RLMetricAccumulator(abc.ABC):
    @abc.abstractmethod
    def track(self, step: Transition) -> None:
        pass

    @abc.abstractmethod
    def calculate(self) -> typing.Dict[str, float]:
        pass


class MeanEpisodeLength(RLMetricAccumulator):
    def __init__(self) -> None:
        super().__init__()
        self.num_steps = 0
        self.num_episodes = 0

    def track(self, step: Transition) -> None:
        _obs, _action, _reward, done, _next_obs = step
        self.num_steps += 1
        self.num_episodes += int(
            done
        )  # FIXME: partial episodes will affect num_steps, but not num_episodes

    def calculate(self) -> typing.Dict[str, float]:
        return {"MeanEpisodeLength": self.num_steps / self.num_episodes}


class MeanEpisodeReward(RLMetricAccumulator):
    """
    FIXME: what should this calculate? (currently it calculates #1)

        1. mean(sum(reward for each reward in episode) for each episode)
        2. mean(mean(reward for each reward in episode) for each episode)

    """

    def __init__(self) -> None:
        super().__init__()
        self.total_reward = 0
        self.num_episodes = 0

    def track(self, step: Transition) -> None:
        _obs, _action, reward, done, _next_obs = step
        self.total_reward += reward
        self.num_episodes += int(
            done
        )  # FIXME: partial episodes will affect total_reward, but not num_episodes

    def calculate(self) -> typing.Dict[str, float]:
        return {"MeanEpisodeReward": self.total_reward / self.num_episodes}


class Chain(RLMetricAccumulator):
    def __init__(self, *accumulators: RLMetricAccumulator) -> None:
        super().__init__()
        self.accumulators = accumulators

    def track(self, step: Transition) -> None:
        for accumulator in self.accumulators:
            accumulator.track(step)

    def calculate(self) -> typing.Dict[str, float]:
        info = {}
        for accumulator in self.accumulators:
            info.update(accumulator.calculate())
        return info


def default_metrics() -> RLMetricAccumulator:
    return Chain(MeanEpisodeReward(), MeanEpisodeLength())

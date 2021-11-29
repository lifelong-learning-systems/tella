import abc
import typing
from ...experiences.rl import MDPTransition


class RLMetricAccumulator(abc.ABC):
    @abc.abstractmethod
    def track(self, transition: MDPTransition) -> None:
        pass

    @abc.abstractmethod
    def calculate(self) -> typing.Dict[str, float]:
        pass


class MeanEpisodeLength(RLMetricAccumulator):
    def __init__(self) -> None:
        super().__init__()
        self.num_transitions = 0
        self.num_episodes = 0

    def track(self, transition: MDPTransition) -> None:
        _obs, _action, _reward, done, _next_obs = transition
        self.num_transitions += 1
        self.num_episodes += int(
            done
        )  # FIXME: partial episodes will affect num_transitions, but not num_episodes

    def calculate(self) -> typing.Dict[str, float]:
        return {"MeanEpisodeLength": self.num_transitions / self.num_episodes}


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

    def track(self, transition: MDPTransition) -> None:
        _obs, _action, reward, done, _next_obs = transition
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

    def track(self, transition: MDPTransition) -> None:
        for accumulator in self.accumulators:
            accumulator.track(transition)

    def calculate(self) -> typing.Dict[str, float]:
        info = {}
        for accumulator in self.accumulators:
            info.update(accumulator.calculate())
        return info


def default_metrics() -> RLMetricAccumulator:
    return Chain(MeanEpisodeReward(), MeanEpisodeLength())

import typing
from tella.curriculum import Transition
from tella.agents import ContinualRLAgent, Observation, Action
import tella


class MinimalRandomAgent(ContinualRLAgent):
    def choose_action(
        self, observations: typing.List[typing.Optional[Observation]]
    ) -> typing.List[typing.Optional[Action]]:
        return [
            None if obs is None else self.action_space.sample() for obs in observations
        ]

    def receive_transition(self, transition: Transition):
        pass


if __name__ == "__main__":
    tella.rl_cli(MinimalRandomAgent)

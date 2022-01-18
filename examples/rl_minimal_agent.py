import logging
import typing

import tella


logger = logging.getLogger("Example Random Agent")


class MinimalRandomAgent(tella.ContinualRLAgent):
    def choose_actions(
        self, observations: typing.List[typing.Optional[tella.Observation]]
    ) -> typing.List[typing.Optional[tella.Action]]:
        return [
            None if obs is None else self.action_space.sample() for obs in observations
        ]

    def receive_transitions(self, transitions: typing.List[typing.Optional[tella.Transition]]):
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tella.rl_cli(MinimalRandomAgent)

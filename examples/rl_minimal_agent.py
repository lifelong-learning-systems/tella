import logging
import typing

import tella


logger = logging.getLogger("Example Random Agent")


class MinimalRandomAgent(tella.ContinualRLAgent):
    def choose_action(
        self, observations: typing.List[typing.Optional[tella.Observation]]
    ) -> typing.List[typing.Optional[tella.Action]]:
        return [
            None if obs is None else self.action_space.sample() for obs in observations
        ]

    def receive_transition(self, transition: tella.Transition):
        pass

    def set_rng_seed(self, seed: int) -> None:
        self.action_space.seed(seed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tella.rl_cli(MinimalRandomAgent)

import typing
import tella
from tella.curriculum.rl_task_variant import StepData
from tella.agents.continual_rl_agent import ContinualRLAgent, Observation, Action


class SimpleRLAgent(ContinualRLAgent):
    def choose_action(
        self, observations: typing.List[typing.Optional[Observation]]
    ) -> typing.List[typing.Optional[Action]]:
        return [
            None if obs is None else self.action_space.sample() for obs in observations
        ]

    def receive_transition(self, step: StepData):
        pass

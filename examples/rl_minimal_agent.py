import typing
from tella.experiences.rl import MDPTransition
from tella.agents.continual_rl_agent import ContinualRLAgent, Observation, Action


class MinimalRandomAgent(ContinualRLAgent):
    def step_observe(
        self, observations: typing.List[typing.Optional[Observation]]
    ) -> typing.List[typing.Optional[Action]]:
        return [
            None if obs is None else self.action_space.sample() for obs in observations
        ]

    def step_transition(self, transition: MDPTransition):
        pass

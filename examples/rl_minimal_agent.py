import typing
from tella.experiences.rl import MDPTransition
from tella.agents.continual_rl_agent import ContinualRLAgent, Observation, Action


class MinimalRandomAgent(ContinualRLAgent):
    def step_observe(
        self, observations: typing.List[Observation]
    ) -> typing.List[Action]:
        return [self.action_space.sample() for obs in observations]

    def step_transition(self, transition: MDPTransition):
        pass

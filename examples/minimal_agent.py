from tella.agent import Agent, Observation, Action


class MinimalRandomAgent(Agent):
    def get_action(self, observation: Observation, **kwargs) -> Action:
        return self.action_space.sample()

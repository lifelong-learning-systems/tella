from tella.agent import Agent, Observation, Action


class MinimalRandomAgent(Agent):
    def step_observe(self, observation: Observation) -> Action:
        return self.action_space.sample()

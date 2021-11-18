from tella.agent import Agent, Observation, Action


class MinimalRandomAgent(Agent):
    def step(self, observation: Observation) -> Action:
        return self.action_space.sample()

    def save_internal_state(self, path: str) -> bool:
        return False

    def restore_internal_state(self, path: str) -> bool:
        return False

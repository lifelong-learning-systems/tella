import typing
import gym
from tella.agent import Agent, Observation, Action


class PrintAgent(Agent):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
    ) -> None:
        super().__init__(observation_space, action_space)
        print(observation_space, action_space)

    def save(self, path: str, **kwargs) -> None:
        print("Saving to", path)

    def load(self, path: str, **kwargs) -> None:
        print("Loading from", path)

    def handle_block_start(self, is_learning_allowed: bool, **kwargs) -> None:
        if is_learning_allowed:
            print("About to start a new learning block")
        else:
            print("About to start a new evaluation block")

    def handle_task_start(
        self, task_info: typing.Optional[typing.Dict[str, typing.Any]], **kwargs
    ) -> None:
        print(f"About to start interacting with a new task: {task_info=}")

    def handle_episode_start(self, **kwargs) -> None:
        print("About to start a new episode")

    def get_action(self, observation: Observation, **kwargs) -> Action:
        print("Return random action")
        return self.action_space.sample()

    def handle_step_result(
        self,
        observation: Observation,
        action: Action,
        reward: float,
        done: bool,
        next_observation: Observation,
        **kwargs,
    ) -> None:
        print(f"Received step result {done=}")

    def handle_episode_end(self, **kwargs) -> None:
        print("Episode just ended")

    def handle_task_end(
        self, task_info: typing.Optional[typing.Dict[str, typing.Any]], **kwargs
    ) -> None:
        print(f"Done interacting with task: {task_info=}")

    def handle_block_end(self, is_learning_allowed: bool, **kwargs) -> None:
        if is_learning_allowed:
            print("Done with the learning block")
        else:
            print("Done with the evaluation block")

    def learning_rollout(self, env: gym.Env, **kwargs) -> None:
        return super().learning_rollout(env, **kwargs)
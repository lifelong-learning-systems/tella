import typing
import gym


class Runner:
    """
    Main runner class of tella. Runs an Agent through a Curriculum.

    .. Intended use:
        agent = ...
        curriculum = ...
        Runner(agent, curriculum).run()
    """

    def __init__(self, agent: "Agent", curriculum: "Curriculum"):
        self.agent = agent
        self.curriculum = curriculum
        self.logger = ...  # NOTE: this would be l2logger

    def run(self):
        for learning_block in self.curriculum.learning_blocks():
            # first run a learning block
            self._run_block(
                learning_block,
                run_episode_fn=self._learning_episode,
                block_params={"is_learning_allowed": True},
            )

            # then run evaluation block
            self._run_block(
                self.curriculum.eval_block(),
                run_episode_fn=self._evaluation_episode,
                block_params={"is_learning_allowed": False},
            )

    def _learning_episode(self, env: gym.Env) -> float:
        # simple pass through to make it consistent with evaluation_episode
        return self.agent.learning_rollout(env)

    def _evaluation_episode(self, env: gym.Env) -> float:
        # explicitly step through environment here for evaluation - we want full control
        total_reward = 0.0
        ...
        return total_reward

    def _run_block(
        self,
        task_blocks: typing.Iterable["TaskBlock"],
        run_episode_fn: typing.Callable[[gym.Env], float],
        block_params: typing.Dict,
    ):
        self.agent.handle_block_start(**block_params)
        for task_block in task_blocks:
            self.agent.handle_task_start({"task_label": ..., "task_params": ...})
            env: gym.Env = task_block.task()
            for i_episode in range(task_block.num_episodes):
                self.agent.handle_episode_start()
                reward = run_episode_fn(env)
                self.agent.handle_episode.end()
                self.logger.log({"reward": reward, ...})  # NOTE: put all l2logger stuff here
            self.agent.handle_task_end({"task_label": ..., "task_params": ...})
        self.agent.handle_block_end(**block_params)

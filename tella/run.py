import typing
import gym


def run(agent: "Agent", curriculum: "Curriculum"):
    # NOTE: notional version of run
    for learning_block in curriculum.learning_blocks():
        # first run a learning block
        _run_block(
            agent,
            learning_block,
            run_episode_fn=_learning_episode,
            block_params={"is_learning_allowed": True},
        )

        # then run evaluation block
        _run_block(
            agent,
            curriculum.eval_block(),
            run_episode_fn=_evaluation_episode,
            block_params={"is_learning_allowed": False},
        )


def _learning_episode(agent: "Agent", env: gym.Env):
    # simple pass through to make it consistent with evaluation_episode
    agent.learning_rollout(env)


def _evaluation_episode(agent: "Agent", env: gym.Env):
    # explicitly step through environment here for evaluation - we want full control
    ...


def _run_block(
    agent: "Agent",
    task_blocks: typing.Iterable["TaskBlock"],
    run_episode_fn: typing.Callable[["Agent", gym.Env], None],
    block_params: typing.Dict,
):
    agent.handle_block_start(**block_params)
    for task_block in task_blocks:
        agent.handle_task_start({"task_label": ..., "task_params": ...})
        env: gym.Env = task_block.task()
        for i_episode in range(task_block.num_episodes):
            agent.handle_episode_start()
            run_episode_fn(agent, env)
            agent.handle_episode.end()
        agent.handle_task_end({"task_label": ..., "task_params": ...})
    agent.handle_block_end(**block_params)

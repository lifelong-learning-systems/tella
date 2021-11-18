import typing
import gym
from tella.agent import Agent
from tella.curriculum import Curriculum, Block, EvalBlock, LearnBlock
from tella.rl_experience import RLEpisodeExperience, ActionFn, Transition


def run_rl(agent: Agent, curriculum: Curriculum[ActionFn, Transition, gym.Env]):
    for block in curriculum.blocks():
        agent.block_start(block.is_learning_allowed())
        for experience in block.experiences():
            env = experience.info()
            agent.task_start(env.observation_space, env.action_space, None, None)
            for obs, action, reward, done, next_obs in experience.generate(agent.step):
                if block.is_learning_allowed():
                    keep_going = agent.step_result(obs, action, reward, done, next_obs)
                else:
                    keep_going = True
                if done:
                    agent.episode_end()
                done = done or keep_going
            agent.task_end(env.observation_space, env.action_space, None, None)
        agent.block_end(block.is_learning_allowed())


class ExampleCurriculum(Curriculum[ActionFn, Transition, gym.Env]):
    def blocks(self) -> typing.Iterable[Block]:
        yield LearnBlock(
            [RLEpisodeExperience(lambda: gym.make("CartPole-v1"), num_episodes=1)]
        )
        yield EvalBlock(
            [RLEpisodeExperience(lambda: gym.make("CartPole-v1"), num_episodes=1)]
        )


if __name__ == "__main__":
    import logging
    from logging_agent import LoggingAgent

    logging.basicConfig(level=logging.INFO)

    curriculum = ExampleCurriculum()
    agent = LoggingAgent()

    run_rl(agent, curriculum)

import gym
from tella.rl_experiment import rl_experiment, _spaces
from .simple_curriculum import SimpleRLCurriculum
from .simple_agent import SimpleRLAgent


def test_space_extraction():
    env = gym.make("CartPole-v1")
    observation_space, action_space = _spaces(SimpleRLCurriculum)
    assert observation_space == env.observation_space
    assert action_space == env.action_space


def test_rl_experiment():
    # TODO what should this test other than being runnable?
    # TODO rl experiment isn't really unit testable since it doesn't have outputs...
    rl_experiment(SimpleRLAgent, SimpleRLCurriculum, 1, 1, "")

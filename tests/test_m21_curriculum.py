import gym.error
import pytest

# m21 curriculum depends on gym_minigrid so skip tests if not available
pytest.importorskip("gym_minigrid")

from tella._curriculums.minigrid.m21 import *


class TestMiniGridReducedActionSpaceWrapper:
    env = SimpleCrossingEnv()

    def test_constructor(self):
        wrapper = MiniGridReducedActionSpaceWrapper(self.env, 3)
        assert wrapper.action_space.n == 3

    def test_too_many_actions(self):
        with pytest.raises(AssertionError):
            MiniGridReducedActionSpaceWrapper(self.env, 10)

    def test_with_continuous_action_space(self):
        # TODO
        pass

    def test_valid_action(self):
        wrapper = MiniGridReducedActionSpaceWrapper(self.env, 3)
        assert 2 == wrapper.action(2)

    def test_invalid_action(self):
        wrapper = MiniGridReducedActionSpaceWrapper(self.env, 3)
        with pytest.raises(gym.error.InvalidAction):
            wrapper.action(3)

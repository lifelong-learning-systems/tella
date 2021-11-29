import typing
import gym
from gym import spaces
import numpy as np


class RandomEnv(gym.Env):
    observation_space = spaces.Box(0, 1, ())
    action_space = spaces.Discrete(2)

    def __init__(self, *args, **kwargs):
        self.rng = np.random.default_rng()

    def seed(self, seed: int) -> typing.List[int]:
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
        self.rng = np.random.default_rng(seed=seed)

    def reset(self):
        return self.observation_space.sample()

    def step(self, action) -> typing.Tuple[typing.Any, float, bool, typing.Dict]:
        assert action in self.action_space
        return (
            self.observation_space.sample(),
            self.rng.random(),
            bool(self.rng.integers(0, 2)),
            {},
        )


class Task1VariantA(RandomEnv):
    pass


class Task1VariantB(RandomEnv):
    pass


class Task2(RandomEnv):
    pass


class Task3Variant1(RandomEnv):
    pass


class Task3Variant2(RandomEnv):
    pass


class Task4(RandomEnv):
    pass


class Task5(RandomEnv):
    pass

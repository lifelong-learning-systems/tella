"""
Copyright © 2021-2022 The Johns Hopkins University Applied Physics Laboratory LLC

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the “Software”), to 
deal in the Software without restriction, including without limitation the 
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import gym
from gym_minigrid.envs import DynamicObstaclesEnv


class CustomDynamicObstaclesEnv(DynamicObstaclesEnv):
    def __init__(
        self, size=8, agent_start_pos=(1, 1), agent_start_dir=0, n_obstacles=4
    ):
        super().__init__(size, agent_start_pos, agent_start_dir, n_obstacles)

        # DynamicObstaclesEnv limits actions, but we want that left for our wrapper
        self.action_space = gym.spaces.Discrete(7)

    def reset(self):
        obs = super().reset()

        # DynamicObstaclesEnv uses the class Ball as obstacles, but those can be picked up.
        #   To prevent this, replace their .can_pickup() method with no_obstacle_pickup().
        def no_obstacle_pickup():
            return False

        for ball in self.obstacles:
            ball.can_pickup = no_obstacle_pickup

        return obs


class CustomDynamicObstaclesS5N2(CustomDynamicObstaclesEnv):
    def __init__(
        self,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
    ):
        super().__init__(5, agent_start_pos, agent_start_dir, 2)


class CustomDynamicObstaclesS6N3(CustomDynamicObstaclesEnv):
    def __init__(
        self,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
    ):
        super().__init__(6, agent_start_pos, agent_start_dir, 3)


class CustomDynamicObstaclesS8N4(CustomDynamicObstaclesEnv):
    def __init__(
        self,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
    ):
        super().__init__(8, agent_start_pos, agent_start_dir, 4)

"""
Copyright © 2021 The Johns Hopkins University Applied Physics Laboratory LLC

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

from gym_minigrid.minigrid import *
from tella._curriculums.minigrid.register import register


class CustomFetchEnv(MiniGridEnv):
    """
    Environment in which the agent has to fetch a specified object
    """

    def __init__(
        self,
        size=8,
        targetType='key',
        targetColor='yellow',
        numTargets=1,
        numObjs=2
    ):
        self.targetType = targetType
        self.targetColor = targetColor
        self.numTargets = numTargets
        self.numObjs = numObjs

        super().__init__(
            grid_size=size,
            max_steps=5*size**2,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height-1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width-1, 0)

        # Initialize valid object types in environment
        types = ['key', 'ball', 'box']

        # Remove target type from list of distractor objects
        types.remove(self.targetType)

        objs = []

        # For each key to be generated
        for _ in range(self.numTargets):
            if self.targetType == 'key':
                obj = Key(self.targetColor)
            elif self.targetType == 'ball':
                obj = Ball(self.targetColor)
            elif self.targetType == 'box':
                obj = Box(self.targetColor)

            self.place_obj(obj)
            objs.append(obj)

        # For each distractor object to be generated
        for _ in range(self.numObjs):
            objType = self._rand_elem(types)
            objColor = self._rand_elem(COLOR_NAMES)

            if objType == 'key':
                obj = Key(objColor)
            if objType == 'ball':
                obj = Ball(objColor)
            elif objType == 'box':
                obj = Box(objColor)

            self.place_obj(obj)
            objs.append(obj)

        # Randomize the player start position and orientation
        self.place_agent()

        descStr = '%s %s' % (self.targetColor, self.targetType)

        # Generate the mission string
        idx = self._rand_int(0, 5)
        if idx == 0:
            self.mission = 'get a %s' % descStr
        elif idx == 1:
            self.mission = 'go get a %s' % descStr
        elif idx == 2:
            self.mission = 'fetch a %s' % descStr
        elif idx == 3:
            self.mission = 'go fetch a %s' % descStr
        elif idx == 4:
            self.mission = 'you must fetch a %s' % descStr
        assert hasattr(self, 'mission')

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)

        if self.carrying:
            if self.carrying.color == self.targetColor and \
               self.carrying.type == self.targetType:
                reward = self._reward()
                done = True
            else:
                reward = 0
                done = True

        return obs, reward, done, info

class CustomFetchEnv5x5T1N2(CustomFetchEnv):
    def __init__(self):
        super().__init__(size=5, numTargets=1, numObjs=2)

class CustomFetchEnv8x8T1N2(CustomFetchEnv):
    def __init__(self):
        super().__init__(size=8, numTargets=1, numObjs=2)

class CustomFetchEnv16x16T2N4(CustomFetchEnv):
    def __init__(self):
        super().__init__(size=16, numTargets=2, numObjs=4)

register(
    id='MiniGrid-CustomFetch-5x5-T1N2-v0',
    entry_point='tella_minigrid.envs:CustomFetchEnv5x5T1N2'
)

register(
    id='MiniGrid-CustomFetch-8x8-T1N2-v0',
    entry_point='tella_minigrid.envs:CustomFetchEnv8x8T1N2'
)

register(
    id='MiniGrid-CustomFetch-16x16-T2N4-v0',
    entry_point='tella_minigrid.envs:CustomFetchEnv16x16T2N4'
)

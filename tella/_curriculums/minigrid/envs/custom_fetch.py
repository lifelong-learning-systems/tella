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

from gym_minigrid.minigrid import *


class CustomFetchEnv(MiniGridEnv):
    """
    Environment in which the agent has to fetch a specified object
    """

    def __init__(
        self,
        size=8,
        target_type="key",
        target_color="yellow",
        num_targets=1,
        num_objs=2,
    ):
        self.target_type = target_type
        self.target_color = target_color
        self.num_targets = num_targets
        self.num_objs = num_objs
        max_steps = 5 * size**2
        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            # Set this to True for maximum speed
            see_through_walls=True,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        # Initialize valid object types in environment
        types = ["key", "ball", "box"]

        # Remove target type from list of distractor objects
        types.remove(self.target_type)

        objs = []

        # For each key to be generated
        for _ in range(self.num_targets):
            if self.target_type == "key":
                obj = Key(self.target_color)
            elif self.target_type == "ball":
                obj = Ball(self.target_color)
            elif self.target_type == "box":
                obj = Box(self.target_color)
            else:
                raise NotImplementedError(f"Unexpected target type {self.target_type}")

            self.place_obj(obj)
            objs.append(obj)

        # For each distractor object to be generated
        for _ in range(self.num_objs):
            obj_type = self._rand_elem(types)
            obj_color = self._rand_elem(COLOR_NAMES)

            if obj_type == "key":
                obj = Key(obj_color)
            elif obj_type == "ball":
                obj = Ball(obj_color)
            elif obj_type == "box":
                obj = Box(obj_color)
            else:
                raise NotImplementedError(f"Unexpected object type {self.target_type}")

            self.place_obj(obj)
            objs.append(obj)

        # Randomize the player start position and orientation
        self.place_agent()

        desc_str = "%s %s" % (self.target_color, self.target_type)

        # Generate the mission string
        idx = self._rand_int(0, 5)
        if idx == 0:
            self.mission = "get a %s" % desc_str
        elif idx == 1:
            self.mission = "go get a %s" % desc_str
        elif idx == 2:
            self.mission = "fetch a %s" % desc_str
        elif idx == 3:
            self.mission = "go fetch a %s" % desc_str
        elif idx == 4:
            self.mission = "you must fetch a %s" % desc_str
        assert hasattr(self, "mission")

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)

        if self.carrying:
            if (
                self.carrying.color == self.target_color
                and self.carrying.type == self.target_type
            ):
                reward = self._reward()
                done = True
            else:
                reward = 0
                done = True

        return obs, reward, done, info


class CustomFetchEnv5x5T1N2(CustomFetchEnv):
    def __init__(self):
        super().__init__(size=5, num_targets=1, num_objs=2)


class CustomFetchEnv8x8T1N2(CustomFetchEnv):
    def __init__(self):
        super().__init__(size=8, num_targets=1, num_objs=2)


class CustomFetchEnv10x10T2N4(CustomFetchEnv):
    def __init__(self):
        super().__init__(size=10, num_targets=2, num_objs=4)


class CustomFetchEnv16x16T2N4(CustomFetchEnv):
    def __init__(self):
        super().__init__(size=16, num_targets=2, num_objs=4)

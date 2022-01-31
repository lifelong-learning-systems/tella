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

from gym_minigrid.roomgrid import RoomGrid


class CustomUnlock(RoomGrid):
    """
    Unlock a door
    """

    def __init__(self, seed=None, num_rows=1, num_cols=2, room_size=6):
        super().__init__(
            num_rows=num_rows,
            num_cols=num_cols,
            room_size=room_size,
            max_steps=8 * room_size**2,
            seed=seed,
        )

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Make sure the two rooms are directly connected by a locked door
        door, _ = self.add_door(0, 0, 0, locked=True)
        # Add a key to unlock the door
        self.add_object(0, 0, "key", door.color)

        self.place_agent(0, 0)

        self.door = door
        self.mission = "open the door"

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if action == self.actions.toggle:
            if self.door.is_open:
                reward = self._reward()
                done = True

        return obs, reward, done, info


class CustomUnlock5x5(CustomUnlock):
    def __init__(self):
        super().__init__(room_size=5)


class CustomUnlock7x7(CustomUnlock):
    def __init__(self):
        super().__init__(room_size=7)


class CustomUnlock8x8(CustomUnlock):
    def __init__(self):
        super().__init__(room_size=8)


class CustomUnlock9x9(CustomUnlock):
    def __init__(self):
        super().__init__(room_size=9)


class CustomUnlock11x11(CustomUnlock):
    def __init__(self):
        super().__init__(room_size=11)


class CustomUnlock16x16(CustomUnlock):
    def __init__(self):
        super().__init__(room_size=16)

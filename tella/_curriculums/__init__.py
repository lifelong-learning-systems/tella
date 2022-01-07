"""
Copyright © 2021 The Johns Hopkins University Applied Physics Laboratory LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from ..curriculum import curriculum_registry
from .cartpole import SimpleCartPoleCurriculum, CartPole1000Curriculum


def load_curriculum_registry():
    """
    Loads some default curriculums into the curriculum registry
    """

    curriculum_registry["SimpleCartPole"] = SimpleCartPoleCurriculum
    curriculum_registry["CartPole-1000"] = CartPole1000Curriculum

    try:
        from .minigrid.simple import (
            SimpleMiniGridCurriculum,
            MiniGridCondensed,
            MiniGridDispersed,
        )
    except ImportError:
        # Skip if gym_minigrid is not installed
        pass
    else:
        curriculum_registry["SimpleMiniGrid"] = SimpleMiniGridCurriculum
        curriculum_registry["MiniGridCondensed"] = MiniGridCondensed
        curriculum_registry["MiniGridDispersed"] = MiniGridDispersed

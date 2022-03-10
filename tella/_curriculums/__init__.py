"""
Copyright © 2021-2022 The Johns Hopkins University Applied Physics Laboratory LLC

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
import logging

from tella._curriculums.minigrid.simple import SimpleStepMiniGridCurriculum

from ..curriculum import curriculum_registry
from .cartpole import SimpleCartPoleCurriculum, CartPole1000Curriculum

logger = logging.getLogger(__name__)


def load_curriculum_registry():
    """
    Loads some default curriculums into the curriculum registry
    """

    curriculum_registry["SimpleCartPole"] = SimpleCartPoleCurriculum
    curriculum_registry["CartPole-1000"] = CartPole1000Curriculum

    try:
        from .minigrid import (
            SimpleMiniGridCurriculum,
            SimpleStepMiniGridCurriculum,
            MiniGridCondensed,
            MiniGridDispersed,
            MiniGridSimpleCrossingS9N1,
            MiniGridSimpleCrossingS9N2,
            MiniGridSimpleCrossingS9N3,
            MiniGridDistShiftR2,
            MiniGridDistShiftR5,
            MiniGridDistShiftR3,
            MiniGridDynObstaclesS6N1,
            MiniGridDynObstaclesS8N2,
            MiniGridDynObstaclesS10N3,
            MiniGridCustomFetchS5T1N2,
            MiniGridCustomFetchS8T1N2,
            MiniGridCustomFetchS10T2N4,
            MiniGridCustomUnlockS5,
            MiniGridCustomUnlockS7,
            MiniGridCustomUnlockS9,
            MiniGridDoorKeyS5,
            MiniGridDoorKeyS6,
            MiniGridDoorKeyS7,
        )
    except ImportError:
        logger.info(
            "Unable to load minigrid curriculums because gym_minigrid is not installed"
        )
    else:
        curriculum_registry["SimpleMiniGrid"] = SimpleMiniGridCurriculum
        curriculum_registry["SimpleStepMiniGrid"] = SimpleStepMiniGridCurriculum
        curriculum_registry["MiniGridCondensed"] = MiniGridCondensed
        curriculum_registry["MiniGridDispersed"] = MiniGridDispersed
        curriculum_registry["MiniGridSimpleCrossingS9N1"] = MiniGridSimpleCrossingS9N1
        curriculum_registry["MiniGridSimpleCrossingS9N2"] = MiniGridSimpleCrossingS9N2
        curriculum_registry["MiniGridSimpleCrossingS9N3"] = MiniGridSimpleCrossingS9N3
        curriculum_registry["MiniGridDistShiftR2"] = MiniGridDistShiftR2
        curriculum_registry["MiniGridDistShiftR5"] = MiniGridDistShiftR5
        curriculum_registry["MiniGridDistShiftR3"] = MiniGridDistShiftR3
        curriculum_registry["MiniGridDynObstaclesS6N1"] = MiniGridDynObstaclesS6N1
        curriculum_registry["MiniGridDynObstaclesS8N2"] = MiniGridDynObstaclesS8N2
        curriculum_registry["MiniGridDynObstaclesS10N3"] = MiniGridDynObstaclesS10N3
        curriculum_registry["MiniGridCustomFetchS5T1N2"] = MiniGridCustomFetchS5T1N2
        curriculum_registry["MiniGridCustomFetchS8T1N2"] = MiniGridCustomFetchS8T1N2
        curriculum_registry["MiniGridCustomFetchS10T2N4"] = MiniGridCustomFetchS10T2N4
        curriculum_registry["MiniGridCustomUnlockS5"] = MiniGridCustomUnlockS5
        curriculum_registry["MiniGridCustomUnlockS7"] = MiniGridCustomUnlockS7
        curriculum_registry["MiniGridCustomUnlockS9"] = MiniGridCustomUnlockS9
        curriculum_registry["MiniGridDoorKeyS5"] = MiniGridDoorKeyS5
        curriculum_registry["MiniGridDoorKeyS6"] = MiniGridDoorKeyS6
        curriculum_registry["MiniGridDoorKeyS7"] = MiniGridDoorKeyS7

    try:
        from .atari import (
            AtariCurriculum,
            BreakoutAndPong,
        )
    except ImportError:
        logger.info("Unable to load Atari curriculums")
    else:
        curriculum_registry["AllAtariGames"] = AtariCurriculum
        curriculum_registry["BreakoutAndPong"] = BreakoutAndPong

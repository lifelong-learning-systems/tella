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

from .simple import SimpleMiniGridCurriculum, SimpleStepMiniGridCurriculum
from .m21 import (
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

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

import pytest
from tella.curriculum import curriculum_registry, validate_curriculum


def test_contents():
    num_expected = 2
    try:
        import gym_minigrid
    except ImportError:
        pass
    else:
        num_expected += 22
    try:
        import gym.envs.atari
    except ImportError:
        pass
    else:
        num_expected += 2
    assert len(curriculum_registry) == num_expected


@pytest.mark.parametrize("curriculum_name", list(curriculum_registry.keys()))
def test_validate(curriculum_name):
    curriculum_cls = curriculum_registry[curriculum_name]
    curriculum = curriculum_cls(0)
    validate_curriculum(curriculum)


@pytest.mark.parametrize("curriculum_name", list(curriculum_registry.keys()))
def test_copy(curriculum_name):
    curriculum_cls = curriculum_registry[curriculum_name]
    curriculum = curriculum_cls(0)
    curriculum.copy()

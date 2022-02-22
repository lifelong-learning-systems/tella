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

import argparse
from unittest.mock import patch

import pytest

from tella.cli import rl_cli
from .simple_curriculum import SimpleRLCurriculum
from .simple_agent import SimpleRLAgent


@patch(
    "argparse.ArgumentParser.parse_args",
    return_value=argparse.Namespace(
        num_parallel_envs=1,
        num_lifetimes=1,
        log_dir="logs",
        render=False,
        agent_seed=None,
        curriculum_seed=None,
        agent_config=None,
        curriculum_config=None,
        lifetime_idx=0,
    ),
)
def test_no_args(p, tmpdir):
    tmpdir.chdir()
    # TODO what should this test other than being runnable?
    rl_cli(SimpleRLAgent, SimpleRLCurriculum)


@patch(
    "argparse.ArgumentParser.parse_args",
    return_value=argparse.Namespace(
        num_parallel_envs=1,
        num_lifetimes=2,
        log_dir="logs",
        render=False,
        agent_seed=None,
        curriculum_seed=None,
        agent_config=None,
        curriculum_config=None,
        lifetime_idx=0,
    ),
)
def test_num_lifetimes(p, tmpdir):
    tmpdir.chdir()
    # TODO what should this test other than being runnable?
    rl_cli(SimpleRLAgent, SimpleRLCurriculum)


@patch(
    "argparse.ArgumentParser.parse_args",
    return_value=argparse.Namespace(
        num_parallel_envs=2,
        num_lifetimes=1,
        log_dir="logs",
        render=False,
        agent_seed=None,
        curriculum_seed=None,
        agent_config=None,
        curriculum_config=None,
        lifetime_idx=0,
    ),
)
def test_num_parallel_envs(p, tmpdir):
    tmpdir.chdir()
    # TODO what should this test other than being runnable?
    rl_cli(SimpleRLAgent, SimpleRLCurriculum)


@patch(
    "argparse.ArgumentParser.parse_args",
    return_value=argparse.Namespace(
        num_parallel_envs=1,
        num_lifetimes=2,
        log_dir="logs",
        curriculum="invalid",
        render=False,
        agent_seed=None,
        curriculum_seed=None,
        agent_config=None,
        lifetime_idx=0,
    ),
)
def test_invalid_curriculum_name(p, tmpdir):
    tmpdir.chdir()
    with pytest.raises(RuntimeError):
        rl_cli(SimpleRLAgent)


@patch(
    "argparse.ArgumentParser.parse_args",
    return_value=argparse.Namespace(
        num_parallel_envs=1,
        num_lifetimes=2,
        log_dir="logs",
        render=False,
        agent_seed=None,
        curriculum_seed=None,
        agent_config=None,
        curriculum_config=None,
        lifetime_idx=0,
    ),
)
@patch("gym.envs.classic_control.CartPoleEnv.render")
def test_no_render(render_patch, argparse_patch, tmpdir):
    tmpdir.chdir()
    rl_cli(SimpleRLAgent, SimpleRLCurriculum)
    assert not render_patch.called


@patch(
    "argparse.ArgumentParser.parse_args",
    return_value=argparse.Namespace(
        num_parallel_envs=1,
        num_lifetimes=2,
        log_dir="logs",
        render=True,
        agent_seed=None,
        curriculum_seed=None,
        agent_config=None,
        curriculum_config=None,
        lifetime_idx=0,
    ),
)
@patch("gym.envs.classic_control.CartPoleEnv.render")
def test_renders(render_patch, argparse_patch, tmpdir):
    tmpdir.chdir()
    rl_cli(SimpleRLAgent, SimpleRLCurriculum)
    assert render_patch.called


@patch(
    "argparse.ArgumentParser.parse_args",
    return_value=argparse.Namespace(
        num_parallel_envs=1,
        num_lifetimes=1,
        log_dir="logs",
        render=False,
        agent_seed=None,
        curriculum_seed=None,
        agent_config="test",
        curriculum_config=None,
        lifetime_idx=0,
    ),
)
def test_agent_config(p, tmpdir):
    class AgentFactory:
        def __init__(self):
            self.agent = None

        def create(self, *args, **kwargs):
            self.agent = SimpleRLAgent(*args, **kwargs)
            return self.agent

    tmpdir.chdir()
    af = AgentFactory()
    rl_cli(af.create, SimpleRLCurriculum)
    assert af.agent.config_file == "test"

import argparse
from unittest.mock import patch

import pytest

from tella.cli import rl_cli
from .simple_curriculum import SimpleRLCurriculum
from .simple_agent import SimpleRLAgent


@patch(
    "argparse.ArgumentParser.parse_args",
    return_value=argparse.Namespace(
        num_parallel_envs=1, num_lifetimes=1, log_dir="logs", render=False
    ),
)
def test_no_args(p, tmpdir):
    tmpdir.chdir()
    # TODO what should this test other than being runnable?
    rl_cli(SimpleRLAgent, SimpleRLCurriculum)


@patch(
    "argparse.ArgumentParser.parse_args",
    return_value=argparse.Namespace(
        num_parallel_envs=1, num_lifetimes=2, log_dir="logs", render=False
    ),
)
def test_num_lifetimes(p, tmpdir):
    tmpdir.chdir()
    # TODO what should this test other than being runnable?
    rl_cli(SimpleRLAgent, SimpleRLCurriculum)


@patch(
    "argparse.ArgumentParser.parse_args",
    return_value=argparse.Namespace(
        num_parallel_envs=2, num_lifetimes=1, log_dir="logs", render=False
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
    ),
)
def test_invalid_curriculum_name(p, tmpdir):
    tmpdir.chdir()
    with pytest.raises(RuntimeError):
        rl_cli(SimpleRLAgent)


@patch(
    "argparse.ArgumentParser.parse_args",
    return_value=argparse.Namespace(
        num_parallel_envs=1, num_lifetimes=2, log_dir="logs", render=False
    ),
)
@patch("tella.env.L2LoggerEnv.render")
def test_no_render(render_patch, argparse_patch, tmpdir):
    tmpdir.chdir()
    rl_cli(SimpleRLAgent, SimpleRLCurriculum)
    assert not render_patch.called


@patch(
    "argparse.ArgumentParser.parse_args",
    return_value=argparse.Namespace(
        num_parallel_envs=1, num_lifetimes=2, log_dir="logs", render=True
    ),
)
@patch("tella.env.L2LoggerEnv.render")
def test_renders(render_patch, argparse_patch, tmpdir):
    tmpdir.chdir()
    rl_cli(SimpleRLAgent, SimpleRLCurriculum)
    assert render_patch.called

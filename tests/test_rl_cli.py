import argparse
from unittest.mock import patch

import pytest

from tella.rl_cli import rl_cli
from .simple_curriculum import SimpleRLCurriculum
from .simple_agent import SimpleRLAgent


@patch(
    "argparse.ArgumentParser.parse_args",
    return_value=argparse.Namespace(num_parallel_envs=1, num_lifetimes=1, log_dir=""),
)
def test_no_args(p):
    # TODO what should this test other than being runnable?
    rl_cli(SimpleRLAgent, SimpleRLCurriculum)


@patch(
    "argparse.ArgumentParser.parse_args",
    return_value=argparse.Namespace(num_parallel_envs=1, num_lifetimes=2, log_dir=""),
)
def test_num_lifetimes(p):
    # TODO what should this test other than being runnable?
    rl_cli(SimpleRLAgent, SimpleRLCurriculum)


@patch(
    "argparse.ArgumentParser.parse_args",
    return_value=argparse.Namespace(num_parallel_envs=2, num_lifetimes=1, log_dir=""),
)
def test_num_parallel_envs(p):
    # TODO what should this test other than being runnable?
    rl_cli(SimpleRLAgent, SimpleRLCurriculum)


@patch(
    "argparse.ArgumentParser.parse_args",
    return_value=argparse.Namespace(num_parallel_envs=1, num_lifetimes=2, log_dir="", curriculum="invalid"),
)
def test_invalid_curriculum_name(p):
    with pytest.raises(RuntimeError):
        rl_cli(SimpleRLAgent)

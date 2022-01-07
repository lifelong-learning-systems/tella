import argparse
from unittest.mock import patch
import typing
import csv
import gym
from tella.experiment import rl_experiment, _spaces, run
from l2logger.validate import run as l2logger_validate
from .simple_curriculum import SimpleRLCurriculum
from .simple_agent import SimpleRLAgent


def test_space_extraction():
    env = gym.make("CartPole-v1")
    observation_space, action_space = _spaces(SimpleRLCurriculum)
    assert observation_space == env.observation_space
    assert action_space == env.action_space


def test_rl_experiment(tmpdir):
    # TODO what should this test other than being runnable?
    # TODO rl experiment isn't really unit testable since it doesn't have outputs...
    rl_experiment(SimpleRLAgent, SimpleRLCurriculum, 1, 1, tmpdir)


def test_all_event_orders(tmpdir):
    env = gym.make("CartPole-v1")
    agent = SimpleRLAgent(0, env.observation_space, env.action_space, 1)
    curriculum = SimpleRLCurriculum(rng_seed=0)

    run(agent, curriculum, render=False, log_dir=tmpdir)

    # fmt: off
    assert agent.all_events == [
        (agent.block_start, "learn"),
            (agent.task_start, "CartPoleEnv"),
                (agent.task_variant_start, "CartPoleEnv", "Default"),
                    (agent.learn_task_variant, "CartPoleEnv", "Default"),
                (agent.task_variant_end, "CartPoleEnv", "Default"),
                (agent.task_variant_start, "CartPoleEnv", "Variant1"),
                    (agent.learn_task_variant, "CartPoleEnv", "Variant1"),
                (agent.task_variant_end, "CartPoleEnv", "Variant1"),
            (agent.task_end, "CartPoleEnv"),
        (agent.block_end, "learn"),
        (agent.block_start, "eval"),
            (agent.task_start, "CartPoleEnv"),
                (agent.task_variant_start, "CartPoleEnv", "Default"),
                    (agent.eval_task_variant, "CartPoleEnv", "Default"),
                (agent.task_variant_end, "CartPoleEnv", "Default"),
            (agent.task_end, "CartPoleEnv"),
        (agent.block_end, "eval"),
    ]
    # fmt: on


def test_run_l2logger_dir(tmpdir):
    tmpdir.chdir()

    env = gym.make("CartPole-v1")
    agent = SimpleRLAgent(0, env.observation_space, env.action_space, 1)
    curriculum = SimpleRLCurriculum(rng_seed=0)

    run(agent, curriculum, render=False, log_dir="logs")
    assert tmpdir.join("logs").check()


def test_log_directory(tmpdir):
    tmpdir.chdir()

    rl_experiment(SimpleRLAgent, SimpleRLCurriculum, 1, 1, "logs1")
    assert tmpdir.join("logs1").check()

    rl_experiment(SimpleRLAgent, SimpleRLCurriculum, 1, 1, "logs2")
    assert tmpdir.join("logs2").check()


def test_l2logger_directory_structure(tmpdir):
    tmpdir.chdir()

    rl_experiment(SimpleRLAgent, SimpleRLCurriculum, 1, 1, "logs")

    assert tmpdir.join("logs").check()
    assert len(tmpdir.join("logs").listdir()) == 1
    assert tmpdir.join("logs").listdir()[0].basename.startswith("SimpleRLCurriculum")

    run_dir = tmpdir.join("logs").listdir()[0]
    assert run_dir.join("logger_info.json").check()
    assert run_dir.join("scenario_info.json").check()
    assert run_dir.join("worker-default").check()

    worker_dir = run_dir.join("worker-default")
    assert worker_dir.join("0-train").check()
    assert worker_dir.join("1-test").check()

    block_0_dir = worker_dir.join("0-train")
    assert len(block_0_dir.listdir()) == 1
    assert block_0_dir.join("data-log.tsv").check()

    block_1_dir = worker_dir.join("1-test")
    assert len(block_1_dir.listdir()) == 1
    assert block_1_dir.join("data-log.tsv").check()


def test_l2logger_validation(tmpdir):
    tmpdir.chdir()

    rl_experiment(SimpleRLAgent, SimpleRLCurriculum, 1, 1, "logs")

    with patch(
        "argparse.ArgumentParser.parse_args",
        return_value=argparse.Namespace(log_dir=tmpdir.join("logs").listdir()[0]),
    ):
        l2logger_validate()


def test_l2logger_tsv_contents(tmpdir):
    tmpdir.chdir()

    rl_experiment(SimpleRLAgent, SimpleRLCurriculum, 1, 1, "logs")

    run_dir = tmpdir.join("logs").listdir()[0]
    worker_dir = run_dir.join("worker-default")
    block_0_dir = worker_dir.join("0-train")
    block_1_dir = worker_dir.join("1-test")
    block_0_tsv = block_0_dir.join("data-log.tsv")
    block_1_tsv = block_1_dir.join("data-log.tsv")

    with open(block_0_tsv) as fp:
        _verify_tsv(
            fp,
            expected_num_completes=2,
            expected_task_names={"CartPoleEnv_Default", "CartPoleEnv_Variant1"},
        )

    with open(block_1_tsv) as fp:
        _verify_tsv(
            fp,
            expected_num_completes=1,
            expected_task_names={"CartPoleEnv_Default"},
        )


def _verify_tsv(fp, expected_num_completes: int, expected_task_names: typing.Set[str]):
    reader = csv.reader(fp, delimiter="\t")

    header = next(reader)
    assert header == [
        "block_num",
        "exp_num",
        "worker_id",
        "block_type",
        "block_subtype",
        "task_name",
        "task_params",
        "exp_status",
        "timestamp",
        "reward",
    ]
    num_completes = 0
    task_names = set()
    for row in reader:
        assert len(row) == len(header)
        if row[header.index("exp_status")] == "complete":
            num_completes += 1
        task_names.add(row[header.index("task_name")])
    assert num_completes == expected_num_completes
    assert task_names == expected_task_names

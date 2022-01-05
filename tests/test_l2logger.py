import argparse
from unittest.mock import patch
import typing
import csv
from tella.experiment import rl_experiment
from l2logger.validate import run
from .simple_agent import SimpleRLAgent
from .simple_curriculum import SimpleRLCurriculum


def test_l2logger_directory_structure(tmpdir):
    tmpdir.chdir()

    rl_experiment(SimpleRLAgent, SimpleRLCurriculum, 1, 1, "")

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
        run()


def test_l2logger_tsv_contents(tmpdir):
    tmpdir.chdir()

    rl_experiment(SimpleRLAgent, SimpleRLCurriculum, 1, 1, "")

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

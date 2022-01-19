import argparse
import csv
import os
from unittest import mock
from collections import defaultdict
import gym
from tella.experiment import rl_experiment, _spaces, run
from l2logger.validate import run as l2logger_validate
from .simple_curriculum import SimpleRLCurriculum, MultiEpisodeRLCurriculum
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


def test_reproducible_experiment_filestructure(tmpdir):
    tmpdir.chdir()

    rl_experiment(
        SimpleRLAgent,
        SimpleRLCurriculum,
        num_lifetimes=2,
        num_parallel_envs=1,
        log_dir="logs1",
    )
    rl_experiment(
        SimpleRLAgent,
        SimpleRLCurriculum,
        num_lifetimes=2,
        num_parallel_envs=1,
        log_dir="logs2",
    )

    assert len(tmpdir.join("logs1").listdir()) == 2
    assert len(tmpdir.join("logs2").listdir()) == 2

    for lifetime1, lifetime2 in zip(
        sorted(tmpdir.join("logs1").listdir()),
        sorted(tmpdir.join("logs2").listdir()),
    ):
        structure1 = [
            (dirnames, filenames)
            for _, dirnames, filenames in sorted(os.walk(lifetime1))
        ]
        structure2 = [
            (dirnames, filenames)
            for _, dirnames, filenames in sorted(os.walk(lifetime2))
        ]

        assert structure1 == structure2


def test_reproducible_experiment_same_contents(tmpdir):
    tmpdir.chdir()

    rl_experiment(
        SimpleRLAgent,
        SimpleRLCurriculum,
        num_lifetimes=2,
        num_parallel_envs=1,
        log_dir="logs1",
        agent_seed=0,
        curriculum_seed=0,
    )
    rl_experiment(
        SimpleRLAgent,
        SimpleRLCurriculum,
        num_lifetimes=2,
        num_parallel_envs=1,
        log_dir="logs2",
        agent_seed=0,
        curriculum_seed=0,
    )

    assert len(tmpdir.join("logs1").listdir()) == 2
    assert len(tmpdir.join("logs2").listdir()) == 2

    for lifetime1, lifetime2 in zip(
        sorted(tmpdir.join("logs1").listdir()),
        sorted(tmpdir.join("logs2").listdir()),
    ):
        worker1 = lifetime1.join("worker-default")
        worker2 = lifetime2.join("worker-default")

        columns_to_ignore = ["timestamp"]
        for block1, block2 in zip(sorted(worker1.listdir()), sorted(worker2.listdir())):
            assert block1.basename == block2.basename
            with open(block1.join("data-log.tsv")) as fp1, open(
                block2.join("data-log.tsv")
            ) as fp2:
                reader1 = csv.DictReader(fp1, delimiter="\t")
                reader2 = csv.DictReader(fp2, delimiter="\t")
                for row1, row2 in zip(reader1, reader2):
                    assert row1.keys() == row2.keys()
                    for key in [
                        key for key in row1.keys() if key not in columns_to_ignore
                    ]:
                        assert row1[key] == row2[key]


def test_reproducible_experiment_different_contents(tmpdir):
    tmpdir.chdir()

    rl_experiment(SimpleRLAgent, SimpleRLCurriculum, 2, 1, "logs1", 0, 0)
    rl_experiment(SimpleRLAgent, SimpleRLCurriculum, 2, 1, "logs2", 1, 1)

    assert len(tmpdir.join("logs1").listdir()) == 2
    assert len(tmpdir.join("logs2").listdir()) == 2

    for lifetime1, lifetime2 in zip(
        sorted(tmpdir.join("logs1").listdir()),
        sorted(tmpdir.join("logs2").listdir()),
    ):
        worker1 = lifetime1.join("worker-default")
        worker2 = lifetime2.join("worker-default")

        something_different = False

        columns_to_ignore = ["timestamp"]
        for block1, block2 in zip(sorted(worker1.listdir()), sorted(worker2.listdir())):
            assert block1.basename == block2.basename
            with open(block1.join("data-log.tsv")) as fp1, open(
                block2.join("data-log.tsv")
            ) as fp2:
                reader1 = csv.DictReader(fp1, delimiter="\t")
                reader2 = csv.DictReader(fp2, delimiter="\t")
                for row1, row2 in zip(reader1, reader2):
                    assert row1.keys() == row2.keys()
                    for key in [
                        key for key in row1.keys() if key not in columns_to_ignore
                    ]:
                        if row1[key] != row2[key]:
                            something_different = True

        assert something_different


def test_all_event_orders(tmpdir):
    env = gym.make("CartPole-v1")
    agent = SimpleRLAgent(0, env.observation_space, env.action_space, 1)
    curriculum = SimpleRLCurriculum(0)

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
    curriculum = SimpleRLCurriculum(0)

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
    assert len(run_dir.listdir()) == 3
    assert run_dir.join("logger_info.json").check()
    assert run_dir.join("scenario_info.json").check()
    assert run_dir.join("worker-default").check()

    worker_dir = run_dir.join("worker-default")
    assert len(worker_dir.listdir()) == 2
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

    with mock.patch(
        "argparse.ArgumentParser.parse_args",
        return_value=argparse.Namespace(log_dir=tmpdir.join("logs").listdir()[0]),
    ):
        l2logger_validate()


def test_l2logger_validation_single_episode_parallel(tmpdir):
    tmpdir.chdir()

    rl_experiment(SimpleRLAgent, SimpleRLCurriculum, 1, 5, "logs")

    with mock.patch(
        "argparse.ArgumentParser.parse_args",
        return_value=argparse.Namespace(log_dir=tmpdir.join("logs").listdir()[0]),
    ):
        l2logger_validate()


def test_l2logger_validation_multi_episode_parallel(tmpdir):
    tmpdir.chdir()

    rl_experiment(SimpleRLAgent, MultiEpisodeRLCurriculum, 1, 5, "logs")

    with mock.patch(
        "argparse.ArgumentParser.parse_args",
        return_value=argparse.Namespace(log_dir=tmpdir.join("logs").listdir()[0]),
    ):
        l2logger_validate()


@mock.patch("l2logger.l2logger.DataLogger.log_record")
def test_l2logger_tsv_num_episodes(log_record, tmpdir):
    tmpdir.chdir()

    rl_experiment(SimpleRLAgent, SimpleRLCurriculum, 1, 1, "logs")

    assert log_record.call_count > 0
    completes_by_block = defaultdict(int)
    for call in log_record.call_args_list:
        (record,), _kwargs = call
        completes_by_block[record["block_num"]] += int(
            record["exp_status"] == "complete"
        )

    assert len(completes_by_block) == 2
    assert completes_by_block[0] == 2
    assert completes_by_block[1] == 1


@mock.patch("l2logger.l2logger.DataLogger.log_record")
def test_l2logger_tsv_task_names(log_record, tmpdir):
    tmpdir.chdir()

    rl_experiment(SimpleRLAgent, SimpleRLCurriculum, 1, 1, "logs")

    assert log_record.call_count > 0
    task_names_by_block = defaultdict(set)
    for call in log_record.call_args_list:
        (record,), _kwargs = call
        task_names_by_block[record["block_num"]].add(record["task_name"])

    assert len(task_names_by_block) == 2
    assert task_names_by_block[0] == {
        "CartPoleEnv_Default",
        "CartPoleEnv_Variant1",
    }
    assert task_names_by_block[1] == {"CartPoleEnv_Default"}


@mock.patch("l2logger.l2logger.DataLogger.log_record")
def test_l2logger_tsv_exp_status(log_record, tmpdir):
    tmpdir.chdir()

    rl_experiment(
        SimpleRLAgent, SimpleRLCurriculum, 1, 1, "logs", curriculum_seed=0, agent_seed=1
    )

    assert log_record.call_count > 0
    for call in log_record.call_args_list:
        (record,), _kwargs = call
        assert record["exp_status"] == "complete"


@mock.patch("l2logger.l2logger.DataLogger.log_record")
def test_l2logger_tsv_episode_reward(log_record, tmpdir):
    tmpdir.chdir()

    rl_experiment(
        SimpleRLAgent, SimpleRLCurriculum, 1, 1, "logs", curriculum_seed=0, agent_seed=1
    )

    assert log_record.call_count > 0

    expected_rewards = [10.0, 14.0, 18.0]  # magic numbers from the seed
    for call in log_record.call_args_list:
        (record,), _kwargs = call
        assert record["reward"] == expected_rewards.pop(0)


@mock.patch("l2logger.l2logger.DataLogger.log_record")
def test_l2logger_tsv_multi_episode_reward(log_record, tmpdir):
    tmpdir.chdir()

    rl_experiment(
        SimpleRLAgent,
        MultiEpisodeRLCurriculum,
        1,
        1,
        "logs",
        curriculum_seed=0,
        agent_seed=1,
    )

    assert log_record.call_count > 0

    # fmt: off
    # magic numbers from the seeds
    expected_rewards = [10.0, 13.0, 27.0, 12.0, 17.0, 14.0, 35.0, 21.0, 19.0, 14.0, 12.0, 17.0]
    # fmt: on
    for call in log_record.call_args_list:
        (record,), _kwargs = call
        assert record["reward"] == expected_rewards.pop(0)
    assert len(expected_rewards) == 0


@mock.patch("l2logger.l2logger.DataLogger.log_record")
def test_masked_environments_exp_nums(log_record, tmpdir):
    rl_experiment(
        SimpleRLAgent, SimpleRLCurriculum, 1, num_parallel_envs=5, log_dir=tmpdir
    )

    assert log_record.call_count > 0
    exp_nums = set()
    for call in log_record.call_args_list:
        (record,), _kwargs = call
        exp_nums.add(record["exp_num"])
    assert exp_nums == {0, 1, 2}


@mock.patch("l2logger.l2logger.DataLogger.log_record")
def test_masked_environments_exp_nums(log_record, tmpdir):
    rl_experiment(
        SimpleRLAgent, MultiEpisodeRLCurriculum, 1, num_parallel_envs=5, log_dir=tmpdir
    )

    assert log_record.call_count > 0
    exp_nums_by_block = defaultdict(set)
    for call in log_record.call_args_list:
        (record,), _kwargs = call
        exp_nums_by_block[record["block_num"]].add(record["exp_num"])

    assert len(exp_nums_by_block) == 2  # three blocks
    assert exp_nums_by_block[0] == {0, 1, 2, 3, 4, 5, 6, 7, 8}
    assert exp_nums_by_block[1] == {9, 10, 11}


@mock.patch("l2logger.l2logger.DataLogger.log_record")
def test_masked_environments_exp_nums_by_variant(log_record, tmpdir):
    rl_experiment(
        SimpleRLAgent, MultiEpisodeRLCurriculum, 1, num_parallel_envs=5, log_dir=tmpdir
    )

    assert log_record.call_count > 0
    exp_nums_by_variant = defaultdict(set)
    for call in log_record.call_args_list:
        (record,), _kwargs = call
        exp_nums_by_variant[(record["block_num"], record["task_name"])].add(
            record["exp_num"]
        )

    assert len(exp_nums_by_variant) == 3
    assert exp_nums_by_variant[(0, "CartPoleEnv_Default")] == {0, 1, 2, 3, 4}
    assert exp_nums_by_variant[(0, "CartPoleEnv_Variant1")] == {5, 6, 7, 8}
    assert exp_nums_by_variant[(1, "CartPoleEnv_Default")] == {9, 10, 11}


@mock.patch("l2logger.l2logger.DataLogger.log_record")
def test_masked_environments_worker_ids_single(log_record, tmpdir):
    rl_experiment(
        SimpleRLAgent, SimpleRLCurriculum, 1, num_parallel_envs=5, log_dir=tmpdir
    )

    assert log_record.call_count > 0
    worker_ids = set()
    for call in log_record.call_args_list:
        (record,), _kwargs = call
        worker_ids.add(record["worker_id"])
    assert worker_ids == {"worker-0"}


@mock.patch("l2logger.l2logger.DataLogger.log_record")
def test_masked_environments_worker_ids_multiple(log_record, tmpdir):
    rl_experiment(
        SimpleRLAgent, MultiEpisodeRLCurriculum, 1, num_parallel_envs=5, log_dir=tmpdir
    )

    assert log_record.call_count > 0
    workers_by_variant = defaultdict(set)
    for call in log_record.call_args_list:
        (record,), _kwargs = call
        workers_by_variant[(record["block_num"], record["task_name"])].add(
            record["worker_id"]
        )

    assert len(workers_by_variant) == 3
    assert workers_by_variant[(0, "CartPoleEnv_Default")] == {
        "worker-0",
        "worker-1",
        "worker-2",
        "worker-3",
        "worker-4",
    }
    assert workers_by_variant[(0, "CartPoleEnv_Variant1")] == {
        "worker-0",
        "worker-1",
        "worker-2",
        "worker-3",
    }
    assert workers_by_variant[(1, "CartPoleEnv_Default")] == {
        "worker-0",
        "worker-1",
        "worker-2",
    }

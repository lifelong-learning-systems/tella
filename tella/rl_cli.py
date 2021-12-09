import argparse
import logging
import typing
from .rl_experiment import rl_experiment, AgentFactory, CurriculumFactory


logger = logging.getLogger(__name__)


def rl_cli(
    agent_factory: AgentFactory,
    curriculum_factory: typing.Optional[CurriculumFactory] = None,
    *,
    parse_args=None,
) -> None:
    """
    Builds a CLI wrapper around :func:`rl_experiment` to enable running experiments with
    the `agent_factory` passed in producing the agent, and the CLI loading
    in a curriculum from the command line.

    Example: ..

        import tella

        class MyAgent(ContinualRLAgent):
            ...

        if __name__ == "__main__":
            tella.rl_cli(MyAgent)

    :param agent_factory: A function or class producing :class:`ContinualRLAgent`.
    :param curriculum_factory: Optional curriculum factory to support only running
        experiments with a set curriculum
    :param parse_args: **Not intended to be used**, optional arguments passed to
        :meth:`argparse.ArgumentParser.parse_args()`.
    :return: None
    """
    # FIXME: remove after https://github.com/darpa-l2m/tella/issues/57
    if curriculum_factory is None:
        raise NotImplementedError(
            "Loading curriculum from CLI is not supported in this release"
        )

    parser = _build_parser(require_curriculum=curriculum_factory is None)

    args = parser.parse_args(args=parse_args)

    if curriculum_factory is None:
        # FIXME: load in curriculum https://github.com/darpa-l2m/tella/issues/57
        assert False
        curriculum_factory = ...

    rl_experiment(
        agent_factory,
        curriculum_factory,
        num_runs=args.num_runs,
        num_cores=args.num_cores,
        log_dir=args.log_dir,
    )


def _build_parser(require_curriculum: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--num-runs",
        default=1,
        type=int,
        help="Number of times to run agent through the curriculum",
    )
    parser.add_argument(
        "--num-cores", default=1, type=int, help="Number of cores to use."
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="The base seed to pass to the curriculum constructor.",
    )
    parser.add_argument(
        "--log-dir",
        default="./logs",
        type=str,
        help="The root directory for the l2logger logs produced",
    )
    if require_curriculum:
        assert False
        # FIXME: argument to support loading curriculum https://github.com/darpa-l2m/tella/issues/57
        parser.add_argument("curriculum_path", type=str, help="Path to curriculum fil")
    return parser

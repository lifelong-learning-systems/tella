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

import argparse
import logging
import typing
from .curriculum import curriculum_registry
from .experiment import rl_experiment, AgentFactory, CurriculumFactory
from .agents import ContinualRLAgent


logger = logging.getLogger(__name__)


def rl_cli(
    agent_factory: AgentFactory,
    curriculum_factory: typing.Optional[CurriculumFactory] = None,
) -> None:
    """
    Builds a CLI wrapper around :func:`tella.experiment.rl_experiment()`
    to enable running experiments with the agent produced by the provided
    :class:`AgentFactory <tella.experiment.AgentFactory>`. The experiment's
    curriculum is either provided here or else loaded as a command line argument.

    Example::

        import tella

        class MyAgent(ContinualRLAgent):
            ...

        if __name__ == "__main__":
            tella.rl_cli(MyAgent)

    :param agent_factory: A function or class producing a :class:`ContinualRLAgent <tella.agents.ContinualRLAgent>`.
    :param curriculum_factory: Optional curriculum factory producing an
        :class:`AbstractCurriculum <tella.curriculum.AbstractCurriculum>` to support running experiments
        with a fixed curriculum. Otherwise, the curriculum is specified on the command line.
    """
    parser = _build_parser(require_curriculum=curriculum_factory is None)

    args = parser.parse_args()

    if not curriculum_factory:
        if args.curriculum not in curriculum_registry:
            raise RuntimeError(f"Unknown curriculum {args.curriculum}")
        curriculum_factory = curriculum_registry[args.curriculum]

    rl_experiment(
        agent_factory,
        curriculum_factory,
        num_lifetimes=args.num_lifetimes,
        num_parallel_envs=args.num_parallel_envs,
        log_dir=args.log_dir,
        lifetime_idx=args.lifetime_idx,
        agent_seed=args.agent_seed,
        curriculum_seed=args.curriculum_seed,
        render=args.render,
        agent_config=args.agent_config,
        curriculum_config=args.curriculum_config,
    )


class DeprecateAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        print(f"Warning: Argument {self.option_strings} is deprecated and unused.")


def _build_parser(require_curriculum: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--lifetime-idx",
        default=0,
        type=int,
        help="The index, starting at zero, of the first lifetime to run. "
        "Use this to skip lifetimes or run a specific lifetime other than the first.",
    )
    parser.add_argument(
        "--num-lifetimes",
        default=1,
        type=int,
        help="Number of lifetimes to execute.",
    )
    parser.add_argument(
        "--num-parallel-envs",
        default=1,
        type=int,
        help="Number of environments to run in parallel inside of task variant blocks. "
        "This enables the use of multiple CPUs at the same time for running environment logic,"
        " via vectorized environments.",
    )
    parser.add_argument(
        "--log-dir",
        default="./logs",
        type=str,
        help="The root directory for the l2logger logs produced.",
    )
    parser.add_argument(
        "--render", action="store_true", help="Whether to render the environment"
    )
    parser.add_argument(
        "--seed",
        action=DeprecateAction,
        help="replaced by --agent-seed and --curriculum-seed",
    )
    parser.add_argument(
        "--agent-seed",
        default=None,
        type=int,
        help="The agent rng seed to use for reproducibility.",
    )
    parser.add_argument(
        "--curriculum-seed",
        default=None,
        type=int,
        help="The curriculum rng seed to use for reproducibility.",
    )
    parser.add_argument(
        "--agent-config",
        default=None,
        type=str,
        help="Optional path to agent config file.",
    )
    parser.add_argument(
        "--curriculum-config",
        default=None,
        type=str,
        help="Optional path to curriculum config file.",
    )
    if require_curriculum:
        parser.add_argument(
            "--curriculum",
            required=True,
            type=str,
            choices=curriculum_registry,
            metavar="CURRICULUM_NAME",
            help="Curriculum name for registry lookup. Registered options are: "
            + ", ".join(curriculum_registry),
        )
    return parser

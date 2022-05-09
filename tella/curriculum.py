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

import abc
import functools
import inspect
import itertools
import sys
import typing
import warnings

import yaml
import numpy as np
import gym

# key: curriculum name, value: AbstractCurriculum class object or factory
curriculum_registry = {}


class ValidationError(ValueError):
    """Raised when there is a problem with a curriculum."""

    pass


class TaskVariant:
    """
    A TaskVariant represents a fixed number of steps or episodes in a single type of :class:`gym.Env`.
    """

    def __init__(
        self,
        task_cls: typing.Type[gym.Env],
        *,
        rng_seed: int,
        params: typing.Optional[typing.Dict] = None,
        task_label: typing.Optional[str] = None,
        variant_label: typing.Optional[str] = "Default",
        num_episodes: typing.Optional[int] = None,
        num_steps: typing.Optional[int] = None,
    ) -> None:
        """
        :param task_cls: the :class:`gym.Env` of this task variant
        :param rng_seed: An integer seed used to repeatably instantiate the environment
        :param params: An optional dict of parameters to be passed as constructor arguments to the environment.
        :param task_label: The name of the task which describes this environment.
        :param variant_label: The name of the variant (of task_label) which describes this environment.
        :param num_episodes: The length limit of this experience in number of episodes.
        :param num_steps: The length limit of this experience in number of steps.

        :raises ValidationError: if neither `num_episodes` or `num_steps` is provided.
        :raises ValidationError: if both `num_episodes` and `num_steps` are provided.
        """
        if params is None:
            params = {}
        if task_label is None:
            task_label = task_cls.__name__
        self.task_cls = task_cls
        self.params = params
        self.task_label = task_label
        self.variant_label = variant_label
        self.rng_seed = rng_seed
        if num_episodes is None and num_steps is None:
            raise ValidationError("Neither num_episodes nor num_steps provided")
        if num_episodes is not None and num_steps is not None:
            raise ValidationError("Both num_episodes and num_steps provided")
        self.num_episodes = num_episodes
        self.num_steps = num_steps

    def validate(self) -> None:
        """
        A method to validate that the experience is set up properly.

        :raises ValidationError: if the experience is not set up properly.
        """

        validate_params(self.task_cls, list(self.params.keys()))

    def make_env(self) -> gym.Env:
        """
        Initializes the gym environment object
        """
        return self.task_cls(**self.params)


class TaskBlock:
    """
    A TaskBlock represents a sequence of one or more :class:`TaskVariant` which all share the same general task.
    """

    def __init__(
        self, task_label: str, task_variants: typing.Iterable[TaskVariant]
    ) -> None:
        """
        :param task_label: The name of the task which describes the environment for all contained variants.
        :param task_variants: A sequence of one or more :class:`TaskVariant`
        """
        super().__init__()
        self.task_label = task_label
        self._task_variants = task_variants

    def task_variants(self) -> typing.Iterable[TaskVariant]:
        """
        :return: An Iterable of :class:`TaskVariant`.
        """

        self._task_variants, task_variants = itertools.tee(self._task_variants, 2)
        return task_variants


class Block:
    """
    Represents a sequence of one or more :class:`TaskBlock`
    """

    @property
    @abc.abstractmethod
    def is_learning_allowed(self) -> bool:
        """
        :return: Bool indicating if this block is intended for learning or evaluation
        """
        raise NotImplementedError

    def __init__(self, task_blocks: typing.Iterable[TaskBlock]) -> None:
        """
        :param task_blocks: A sequence of one or more :class:`TaskBlock`
        """
        self._task_blocks = task_blocks

    def task_blocks(self) -> typing.Iterable[TaskBlock]:
        """
        :return: An Iterable of :class:`TaskBlock`.
        """

        self._task_blocks, task_blocks = itertools.tee(self._task_blocks, 2)
        return task_blocks


class LearnBlock(Block):
    """
    A :class:`Block` where the data can be used for learning.
    """

    is_learning_allowed = True


class EvalBlock(Block):
    """
    A :class:`Block` where the data can NOT be used for learning.
    """

    is_learning_allowed = False


class AbstractCurriculum:
    """
    Represents a lifelong/continual learning curriculum. A curriculum is simply
    a sequence of one or more :class:`Block`.
    """

    def __init__(self, rng_seed: int, config_file: typing.Optional[str] = None):
        """
        :param rng_seed: The seed to be used in setting random number generators.
        :param config_file: Path to a config file for the curriculum or None if no config.
        """
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)

        if config_file is not None:
            with open(config_file) as file:
                self.config = yaml.safe_load(file)
        else:
            self.config = {}
        self.config_file = config_file

    def copy(self) -> "AbstractCurriculum":
        """
        :return: A new instance of this curriculum, initialized with the same arguments.

        .. NOTE::
            Curriculum authors will need to overwrite this method for subclasses with different arguments.
        """
        return self.__class__(self.rng_seed, self.config_file)

    @abc.abstractmethod
    def learn_blocks_and_eval_blocks(self) -> typing.Iterable[Block]:
        """
        Generate the learning and eval blocks of this curriculum.

        :return: An Iterable of :class:`Block`.
        """
        raise NotImplementedError

    def validate(self) -> None:
        """
        A method to validate that the curriculum is set up properly.
        This is an optional capability for curriculum authors to implement.

        :raises ValidationError: if the curriculum is not set up properly.
        """
        pass


class InterleavedEvalCurriculum(AbstractCurriculum):
    """
    A common curriculum format where a single evaluation block is repeated
    before, between, and after the curriculum's learning blocks.

    This class implements :meth:`AbstractCurriculum.learn_blocks_and_eval_blocks()`,
    and expects the user to implement two new methods:

        1. learn_blocks(), which returns the sequence of :class:`LearnBlock`.
        2. eval_block(), which returns the single :class:`EvalBlock` to be
           interleaved between each :class:`LearnBlock`.

    """

    def __init__(self, rng_seed: int, config_file: typing.Optional[str] = None):
        super().__init__(rng_seed, config_file)
        # Also save a fixed eval_rng_seed so that eval environments are the same in each block
        self.eval_rng_seed = self.rng.bit_generator.random_raw()

    @abc.abstractmethod
    def learn_blocks(self) -> typing.Iterable[LearnBlock]:
        """
        :return: An iterable of :class:`LearnBlock`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def eval_block(self) -> EvalBlock:
        """
        :return: The single :class:`EvalBlock` to interleave between each
            individual :class:`LearnBlock` returned from
            :meth:`InterleavedEvalCurriculum.learn_blocks`.
        """
        raise NotImplementedError

    def learn_blocks_and_eval_blocks(self) -> typing.Iterable[Block]:
        yield self.eval_block()
        for block in self.learn_blocks():
            yield block
            yield self.eval_block()


def split_task_variants(
    task_variants: typing.Iterable[TaskVariant],
) -> typing.Iterable[TaskBlock]:
    """
    Divides task variants into one or more blocks of matching tasks

    :param task_variants: The iterable of :class:`TaskVariant` to be placed into task blocks.
    :return: An iterable of one or more :class:`TaskBlock` which contain the provided task variants.
    """
    for task_label, variants_in_block in itertools.groupby(
        task_variants, key=lambda task: task.task_label
    ):
        yield TaskBlock(task_label, variants_in_block)


def simple_learn_block(task_variants: typing.Iterable[TaskVariant]) -> LearnBlock:
    """
    Constructs a :class:`LearnBlock` with the task variants passed in. :class:`TaskBlock` are divided as needed.

    :param task_variants: The iterable of :class:`TaskVariant` to include in the :class:`LearnBlock`.
    :return: A :class:`LearnBlock` with one or more :class:`TaskBlock` which
        contain the provided task variants.
    """
    return LearnBlock(split_task_variants(task_variants))


def simple_eval_block(task_variants: typing.Iterable[TaskVariant]) -> EvalBlock:
    """
    Constructs a :class:`EvalBlock` with the task variants passed in. :class:`TaskBlock` are divided as needed.

    :param task_variants: The iterable of :class:`TaskVariant` to include in the :class:`EvalBlock`.
    :return: A :class:`EvalBlock` with one or more :class:`TaskBlock` which
        contain the provided task variants.
    """
    return EvalBlock(split_task_variants(task_variants))


Observation = typing.TypeVar("Observation")  #: Observation of environment state
Action = typing.TypeVar("Action")  #: Action taken in environment
Reward = float  #: Float reward from environment
Done = bool  #: Bool, True if episode has ended


class Transition(typing.NamedTuple):
    """
    A named tuple containing data from a single step in an MDP:
    (observation, action, reward, done, next_observation)
    """

    observation: Observation
    action: Action
    reward: typing.Optional[Reward]
    done: Done
    next_observation: Observation


def summarize_curriculum(
    curriculum: AbstractCurriculum,
) -> str:
    """
    Generate a detailed string summarizing the contents of the curriculum.

    :return: A string that would print as a formatted outline of this curriculum's contents.
    """

    def maybe_plural(num: int, label: str):
        return f"{num} {label}" + ("" if num == 1 else "s")

    block_summaries = []
    for i_block, block in enumerate(curriculum.learn_blocks_and_eval_blocks()):

        task_summaries = []
        for i_task, task_block in enumerate(block.task_blocks()):

            variant_summaries = []
            for i_variant, task_variant in enumerate(task_block.task_variants()):
                variant_summary = (
                    f"\n\t\t\tTask variant {i_variant+1}, "
                    f"{task_variant.task_label} - {task_variant.variant_label}: "
                    + (
                        f"{maybe_plural(task_variant.num_episodes, 'episode')}."
                        if task_variant.num_episodes is not None
                        else f"{maybe_plural(task_variant.num_steps, 'step')}."
                    )
                )
                variant_summaries.append(variant_summary)

            task_summary = (
                f"\n\t\tTask {i_task+1}, {task_block.task_label}: "
                f"{maybe_plural(len(variant_summaries), 'variant')}"
            )
            task_summaries.append(task_summary + "".join(variant_summaries))

        block_summary = (
            f"\n\n\tBlock {i_block+1}, "
            f"{'learning' if block.is_learning_allowed else 'evaluation'}: "
            f"{maybe_plural(len(task_summaries), 'task')}"
        )
        block_summaries.append(block_summary + "".join(task_summaries))

    curriculum_summary = (
        f"This curriculum has {maybe_plural(len(block_summaries), 'block')}"
        + "".join(block_summaries)
    )

    return curriculum_summary


@functools.lru_cache()
def _env_spaces(env_constructor):
    env = env_constructor()
    spaces = (env.observation_space, env.action_space)
    env.close()
    return spaces


def validate_curriculum(
    curriculum: AbstractCurriculum,
):
    """
    Helper function to do a partial check that task variants are specified
    correctly in the blocks of the :class:`AbstractCurriculum`.

    Uses :meth:`AbstractCurriculum.validate()` if implemented for the curriculum.
    Uses :meth:`TaskVariant.validate()` to check task variants.

    :raises ValidationError: if an invalid parameter is detected.
    :raises ValidationError: if the curriculum contains multiple observation or action spaces.
    :raises ValidationError: if any task block contains multiple tasks.
    :raises ValidationError: if the curriculum, or any block, or any task block is empty.

    :return: None
    """
    curriculum.validate()

    warned_repeat_variants = False
    obs_and_act_spaces = None  # placeholder
    empty_curriculum = True
    for i_block, block in enumerate(curriculum.learn_blocks_and_eval_blocks()):
        empty_curriculum = False

        empty_block = True
        for i_task_block, task_block in enumerate(block.task_blocks()):
            empty_block = False
            task_labels = set()
            variant_labels = set()

            empty_task = True
            previous_variant = None
            for i_task_variant, task_variant in enumerate(task_block.task_variants()):
                empty_task = False
                task_labels.add(task_variant.task_label)
                variant_labels.add(task_variant.variant_label)

                # Validate this individual task variant instance
                try:
                    task_variant.validate()
                except ValidationError as e:
                    raise ValidationError(
                        f"Invalid task variant at block #{i_block}, "
                        f"task block #{i_task_block}, "
                        f"task variant #{i_task_variant}."
                    ) from e

                # Warn once if any adjacent task variants are the same
                if (
                    not warned_repeat_variants
                    and i_task_variant
                    and task_variant.variant_label == previous_variant
                ):
                    warnings.warn(
                        f"Multiple task variants share the same variant label "
                        f"'{task_variant.variant_label}' for task '{task_variant.task_label}'."
                    )
                    warned_repeat_variants = True
                previous_variant = task_variant.variant_label

                # Check that all environments use the same observation and action spaces
                observation_space, action_space = _env_spaces(task_variant.task_cls)
                if obs_and_act_spaces is None:
                    obs_and_act_spaces = (observation_space, action_space)
                else:
                    if obs_and_act_spaces != (observation_space, action_space):
                        raise ValidationError(
                            "All environments in a curriculum must use the same observation and action spaces."
                        )

            # Check that task blocks only contain one task
            if len(task_labels) > 1:
                raise ValidationError(
                    f"Block #{i_block}, task block #{i_task_block} had more than 1"
                    f" task label found across all task variants: {task_labels}"
                )

            # Check that no empty blocks are included
            if empty_task:
                raise ValidationError(
                    f"Block #{i_block}, task block #{i_task_block} is empty."
                )
        if empty_block:
            raise ValidationError(f"Block #{i_block} is empty.")
    if empty_curriculum:
        raise ValidationError(f"This curriculum is empty.")


def validate_params(fn: typing.Callable, param_names: typing.List[str]) -> None:
    """
    Determines whether there are missing or invalid names in `param_names` to pass
    to the function `fn`.

    .. WARNING::
        if `fn` has any ``**kwargs``, then all arguments are valid and this method
        won't be able to verify anything.

    :param fn: The callable that will accept the parameters.
    :param param_names: The names of the parameters to check.

    :raises ValidationError: if any of `param_names` are not found in the signature, and there are no ``**kwargs``
    :raises ValidationError: if any of the parameters without defaults in `fn` are not present in `param_names`
    :raises ValidationError: if any `*args` are found
    :raises ValidationError: if any positional only arguments are found (i.e. using /)
    """

    fn_signature = inspect.signature(fn)
    # inspect was broken before Python 3.9 for constructors that inherit from Generic
    # signature came out as *args, **kwds in earlier versions
    # https://github.com/python/cpython/issues/85074
    old_inspect = sys.version_info < (3, 9)

    kwarg_found = False
    expected_fn_names = []
    for param_name, param in fn_signature.parameters.items():
        if param.kind == param.VAR_POSITIONAL:
            if old_inspect:
                # cannot inspect signature due to python bug so hope for the best
                return
            raise ValidationError(
                f"*args not allowed. Found {param_name} in {fn_signature}"
            )
        if param.kind == param.POSITIONAL_ONLY:
            raise ValidationError(
                f"Positional only arguments not allowed. Found {param_name} in {fn_signature}"
            )
        if param.kind == param.VAR_KEYWORD:
            kwarg_found = True
        elif param.default is param.empty:
            expected_fn_names.append(param_name)

    # NOTE: this is checking for presence of name in the fn_signature.
    # NOTE: **kwargs absorb everything!
    invalid_params = []
    for name in param_names:
        if name not in fn_signature.parameters:
            invalid_params.append(name)
    if len(invalid_params) > 0 and not kwarg_found:
        raise ValidationError(
            f"Parameters not accepted: {invalid_params} in {fn_signature}"
        )

    # NOTE: this is checking for parameters that don't have a default specified
    missing_params = []
    for name in expected_fn_names:
        if name not in param_names:
            missing_params.append(name)
    if len(missing_params) > 0:
        raise ValidationError(f"Missing parameters: {missing_params} in {fn_signature}")

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
import inspect
import itertools
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
    A TaskVariant abstractly represents some amount of experience in a single task.

    We represent a TaskVariant as something that takes an input object (`InputType`) and produces
    an generic experience `ExperienceType` object. Additionally, a TaskVariant
    has some information (`InfoType`) associated with it.

    For RL, a TaskVariant can be thought of as taking an agent as InputType,
    producing an Iterable of Step Data as ExperienceType, and giving
    and :class:`gym.Env` as the InfoType.
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
        TODO
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
    TODO
    """

    def __init__(
        self, task_label: str, task_variants: typing.Iterable[TaskVariant]
    ) -> None:
        """
        TODO
        """
        super().__init__()
        self.task_label = task_label
        self._task_variants = task_variants

    def task_variants(self) -> typing.Iterable[TaskVariant]:
        """
        :return: An Iterable of :class:`TaskVariantType`.
        """

        self._task_variants, task_variants = itertools.tee(self._task_variants, 2)
        return task_variants


class Block:
    """
    Represents a sequence of 1 or more :class:`TaskBlock`
    """

    @property
    @abc.abstractmethod
    def is_learning_allowed(self) -> bool:
        """
        :return: Bool indicating if this block is intended for learning or evaluation
        """
        raise NotImplementedError

    def __init__(self, task_blocks: typing.Iterable[TaskBlock]) -> None:
        super().__init__()
        self._task_blocks = task_blocks

    def task_blocks(self) -> typing.Iterable[TaskBlock]:
        """
        :return: An Iterable of Task Blocks
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
    a sequence of :class:`AbstractLearnBlock` and :class:`AbstractEvalBlock`.
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
        :return: A new instance of this curriculum, initialized with the same rng_seed and config_file

        Curriculum authors will need to overwrite this method for subclasses with additional inputs
        """
        return self.__class__(self.rng_seed, self.config_file)

    @abc.abstractmethod
    def learn_blocks_and_eval_blocks(
        self,
    ) -> typing.Iterable[typing.Union[LearnBlock, EvalBlock]]:
        """
        Generate the learning and eval blocks of this curriculum.

        :return: An Iterable of Learn Blocks and Eval Blocks.
        """
        raise NotImplementedError


class InterleavedEvalCurriculum(AbstractCurriculum):
    """
    One possible version of a curriculum where a single evaluation block
    is interleaved between a sequence of learning blocks.

    This class implements :meth:`AbstractCurriculum.learn_blocks_and_eval_blocks()`,
    and expects the user to implement two new methods:

        1. learn_blocks(), which returns the sequence of :class:`AbstractLearnBlock`.
        2. eval_block(), which returns the single :class:`AbstractEvalBlock` to be
           interleaved between each :class:`AbstractLearnBlock`.

    """

    def __init__(self, rng_seed: int, config_file: typing.Optional[str] = None):
        """
        :param rng_seed: The seed to be used in setting random number generators.
        :param config_file: Path to a config file for the curriculum or None if no config.
        """
        super().__init__(rng_seed, config_file)
        # Also save a fixed eval_rng_seed so that eval environments are the same in each block
        self.eval_rng_seed = self.rng.bit_generator.random_raw()

    @abc.abstractmethod
    def learn_blocks(self) -> typing.Iterable[LearnBlock]:
        """
        :return: An iterable of :class:`AbstractLearnBlock`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def eval_block(self) -> EvalBlock:
        """
        :return: The single :class:`AbstractEvalBlock` to interleave between each
            individual :class:`AbstractLearnBlock` returned from
            :meth:`InterleavedEvalCurriculum.learn_blocks`.
        """
        raise NotImplementedError

    def learn_blocks_and_eval_blocks(
        self,
    ) -> typing.Iterable[typing.Union[LearnBlock, EvalBlock]]:
        yield self.eval_block()
        for block in self.learn_blocks():
            yield block
            yield self.eval_block()


def split_task_variants(
    task_variants: typing.Iterable[TaskVariant],
) -> typing.Iterable[TaskBlock]:
    """
    Divides task variants into one or more blocks of matching tasks

    :param task_variants: The iterable of TaskVariantType to be placed into task blocks.
    :return: An iterable of one or more :class:`TaskBlock` which contain the provided `task_variants`.
    """
    for task_label, variants_in_block in itertools.groupby(
        task_variants, key=lambda task: task.task_label
    ):
        yield TaskBlock(task_label, variants_in_block)


def simple_learn_block(
    task_variants: typing.Iterable[TaskVariant],
) -> LearnBlock:
    """
    Constucts a learn block with the task variants passed in. Task blocks are divided as needed.

    :param task_variants: The iterable of TaskVariantType to include in the learn block.
    :return: A :class:`LearnBlock` with one or more :class:`TaskBlock` which
        contain the `task_variants` parameter.
    """
    return LearnBlock(split_task_variants(task_variants))


def simple_eval_block(
    task_variants: typing.Iterable[TaskVariant],
) -> EvalBlock:
    """
    Constucts an eval block with the task variants passed in. Task blocks are divided as needed.

    :param task_variants: The iterable of TaskVariantType to include in the eval block.
    :return: A :class:`EvalBlock` with one or more :class:`TaskBlock` which
        contain the `task_variants` parameter.
    """
    return EvalBlock(split_task_variants(task_variants))


Observation = typing.TypeVar("Observation")
Action = typing.TypeVar("Action")
Reward = float
Done = bool
NextObservation = Observation


class Transition(typing.NamedTuple):
    """
    A named tuple containing data from a single step in an MDP:
    (observation, action, reward, done, next_observation)
    """

    observation: Observation
    action: Action
    reward: typing.Optional[Reward]
    done: Done
    next_observation: NextObservation


ActionFn = typing.Callable[
    [typing.List[typing.Optional[Observation]]], typing.List[typing.Optional[Action]]
]
"""
A function that takes a list of Observations and returns a list of Actions, one
for each observation.
"""


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


def validate_curriculum(
    curriculum: AbstractCurriculum,
):
    """
    Helper function to do a partial check that task variants are specified
    correctly in the blocks of the `curriculum`.

    Uses :meth:`AbstractTaskVariant.validate()` to check task variants.

    :raises ValidationError: if an invalid parameter is detected.
    :raises ValidationError: if the curriculum contains multiple observation or action spaces.
    :raises ValidationError: if any task block contains multiple tasks.
    :raises ValidationError: if the curriculum, or any block, or any task block is empty.

    :return: None
    """
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
                env = task_variant.make_env()
                if isinstance(env, gym.vector.VectorEnv):
                    observation_space = env.single_observation_space
                    action_space = env.single_action_space
                else:
                    observation_space = env.observation_space
                    action_space = env.action_space
                env.close()
                del env
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

    NOTE:
        if `fn` has any ``**kwargs``, then all arguments are valid and this method
        won't be able to verify anything.

    :param fn: The callable that will accept the parameters.
    :param param_names: The names of the parameters to check.

    :raises: a ValidationError if any of `param_names` are not found in the signature, and there are no ``**kwargs``
    :raises: a ValidationError if any of the parameters without defaults in `fn` are not present in `param_names`
    :raises: a ValidationError if any `*args` are found
    :raises: a ValidationError if any positional only arguments are found (i.e. using /)
    """

    fn_signature = inspect.signature(fn)

    kwarg_found = False
    expected_fn_names = []
    for param_name, param in fn_signature.parameters.items():
        if param.kind == param.VAR_POSITIONAL:
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

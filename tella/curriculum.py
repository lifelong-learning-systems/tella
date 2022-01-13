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

import numpy as np
import gym

from .env import L2LoggerEnv

# key: curriculum name, value: AbstractCurriculum class object or factory
curriculum_registry = {}

InputType = typing.TypeVar("InputType")
ExperienceType = typing.TypeVar("ExperienceType")
InfoType = typing.TypeVar("InfoType")


class AbstractTaskVariant(abc.ABC, typing.Generic[InputType, ExperienceType, InfoType]):
    """
    A TaskVariant abstractly represents some amount of experience in a single task.

    We represent a TaskVariant as something that takes an input object (`InputType`) and produces
    an generic experience `ExperienceType` object. Additionally, a TaskVariant
    has some information (`InfoType`) associated with it.

    This representation allows us to represent both RL and SL tasks/experiences.

    For RL, a TaskVariant can be thought of as taking an agent as InputType,
    producing an Iterable of Step Data as ExperienceType, and giving
    and :class:`gym.Env` as the InfoType.

    For SL, a TaskVariant could take a batch size integer as
    InputType, and produce a Batch of image/label data as ExperienceType.
    """

    @abc.abstractmethod
    def validate(self) -> None:
        """
        A method to validate that the experience is set up properly.

        This should raise an Exception if the experience is not set up properly.
        """

    @abc.abstractmethod
    def info(self) -> InfoType:
        """
        :return: The object of type `InfoType` associated with this experience.
        """

    @abc.abstractmethod
    def generate(self, inp: InputType) -> ExperienceType:
        """
        The main method to generate the Experience data.

        :param inp: The object of type InputType.
        :return: The data for the experience.
        """

    @property
    @abc.abstractmethod
    def task_label(self) -> str:
        """
        :return: The task label associated with this task variant. All task variants
            with the same task should have the same task label.
        """

    @property
    @abc.abstractmethod
    def variant_label(self) -> str:
        """
        :return: The variant label associated with this task variant. All task variants
            with the same extrinsic parameters should have the same variant label.
        """


TaskVariantType = typing.TypeVar("TaskVariantType", bound=AbstractTaskVariant)


class AbstractCurriculum(abc.ABC, typing.Generic[TaskVariantType]):
    """
    Represents a lifelong/continual learning curriculum. A curriculum is simply
    a sequence of :class:`AbstractLearnBlock`s and :class:`AbstractEvalBlock`s.
    """

    def __init__(self, rng_seed: int):
        """
        :param rng_seed: The seed to be used in setting random number generators. This should be referenced
            at each call to .learn_blocks_and_eval_blocks()
        """
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)

    def copy(self) -> "AbstractCurriculum":
        """
        :return: A new instance of this curriculum

        Curriculum authors will need to overwrite this method for subclasses with additional inputs
        """
        return self.__class__(self.rng_seed)

    @abc.abstractmethod
    def learn_blocks_and_eval_blocks(
        self,
    ) -> typing.Iterable[
        typing.Union[
            "AbstractLearnBlock[TaskVariantType]", "AbstractEvalBlock[TaskVariantType]"
        ]
    ]:
        """
        Generate the learning and eval blocks of this curriculum.

        If any randomness is used in a curriculum, the random number generator(s) should be seeded based on
        self.rng_seed. This should be reset at each call to .learn_blocks_and_eval_blocks() so that each
        returns identical blocks.

        :return: An Iterable of Learn Blocks and Eval Blocks.
        """


class AbstractLearnBlock(abc.ABC, typing.Generic[TaskVariantType]):
    """
    Represents a sequence of 1 or more :class:`AbstractTaskBlock`, where the
    data can be used for learning.
    """

    @property
    def is_learning_allowed(self) -> bool:
        return True

    @abc.abstractmethod
    def task_blocks(self) -> typing.Iterable["AbstractTaskBlock[TaskVariantType]"]:
        """
        :return: An Iterable of Task Blocks
        """


class AbstractEvalBlock(abc.ABC, typing.Generic[TaskVariantType]):
    """
    Represents a sequence of 1 or more :class:`AbstractTaskBlock`, where the
    data can NOT be used for learning.
    """

    @property
    def is_learning_allowed(self) -> bool:
        return False

    @abc.abstractmethod
    def task_blocks(self) -> typing.Iterable["AbstractTaskBlock[TaskVariantType]"]:
        """
        :return: An Iterable of Task Blocks
        """


class AbstractTaskBlock(abc.ABC, typing.Generic[TaskVariantType]):
    """
    Represents a sequence of 1 or more Task Variants (represented by the
    generic type `TaskVariantType`.)
    """

    def task_variants(self) -> typing.Iterable[TaskVariantType]:
        """
        :return: An Iterable of :class:`TaskVariantType`.
        """

    @property
    @abc.abstractmethod
    def task_label(self) -> str:
        """
        :return: The task label associated with this task variant. All task variants
            with the same task should have the same task label.
        """


class InterleavedEvalCurriculum(AbstractCurriculum[TaskVariantType]):
    """
    One possible version of a curriculum where a single evaluation block
    is interleaved between a sequence of learning blocks.

    This class implements :meth:`Curriculum.blocks()`, and expects the user
    to implement two new methods:

        1. learn_blocks(), which returns the sequence of :class:`LearnBlock`.
        2. eval_block(), which returns the single :class:`EvalBlock` to be
            interleaved between each :class:`LearnBlock`.

    """

    def __init__(self, rng_seed: int):
        """
        :param rng_seed: The seed to be used in setting random number generators. This should be referenced
            at each call to .learn_blocks_and_eval_blocks()
        """
        super().__init__(rng_seed)
        # Also save a fixed eval_rng_seed so that eval environments are the same in each block
        self.eval_rng_seed = self.rng.bit_generator.random_raw()

    @abc.abstractmethod
    def learn_blocks(self) -> typing.Iterable[AbstractLearnBlock[TaskVariantType]]:
        """
        :return: An iterable of :class:`LearnBlock`.
        """

    @abc.abstractmethod
    def eval_block(self) -> AbstractEvalBlock[TaskVariantType]:
        """
        :return: The single :class:`EvalBlock` to interleave between each
            individual :class:`LearnBlock` returned from
            :meth:`InterleavedEvalCurriculum.learn_blocks`.
        """

    def learn_blocks_and_eval_blocks(
        self,
    ) -> typing.Iterable[
        typing.Union[
            "AbstractLearnBlock[TaskVariantType]", "AbstractEvalBlock[TaskVariantType]"
        ]
    ]:
        yield self.eval_block()
        for block in self.learn_blocks():
            yield block
            yield self.eval_block()


class TaskBlock(AbstractTaskBlock):
    """
    A simple subclass of :class:`AbstractTaskBlock` that accepts the task variants
    in the constructor.
    """

    def __init__(
        self, task_label: str, task_variants: typing.Iterable[TaskVariantType]
    ) -> None:
        super().__init__()
        self._task_label = task_label
        self._task_variants = task_variants

    def task_variants(self) -> typing.Iterable[TaskVariantType]:
        self._task_variants, task_variants = itertools.tee(self._task_variants, 2)
        return task_variants

    @property
    def task_label(self) -> str:
        return self._task_label


class LearnBlock(AbstractLearnBlock):
    """
    A simple subclass of :class:`AbstractLearnBlock` that accepts the task blocks
    in the constructor.
    """

    def __init__(self, task_blocks: typing.Iterable[TaskBlock]) -> None:
        super().__init__()
        self._task_blocks = task_blocks

    def task_blocks(self) -> typing.Iterable["AbstractTaskBlock[TaskVariantType]"]:
        self._task_blocks, task_blocks = itertools.tee(self._task_blocks, 2)
        return task_blocks


class EvalBlock(AbstractEvalBlock):
    """
    A simple subclass of :class:`AbstractEvalBlock` that accepts the task blocks
    in the constructor.
    """

    def __init__(self, task_blocks: typing.Iterable[TaskBlock]) -> None:
        super().__init__()
        self._task_blocks = task_blocks

    def task_blocks(self) -> typing.Iterable["AbstractTaskBlock[TaskVariantType]"]:
        self._task_blocks, task_blocks = itertools.tee(self._task_blocks, 2)
        return task_blocks


def split_task_variants(
    task_variants: typing.Iterable[TaskVariantType],
) -> typing.Iterable[TaskBlock]:
    """
    Divides task variants into one or more blocks of matching tasks

    :param task_variants: The iterable of TaskVariantType to be placed into task blocks.
    :return: An iterable of one or more :class:`TaskBlock`s which contain the provided `task_variants`.
    """
    for task_label, variants_in_block in itertools.groupby(
        task_variants, key=lambda task: task.task_label
    ):
        yield TaskBlock(task_label, variants_in_block)


def simple_learn_block(
    task_variants: typing.Iterable[TaskVariantType],
) -> AbstractLearnBlock[TaskVariantType]:
    """
    Constucts a learn block with the task variants passed in. Task blocks are divided as needed.

    :param task_variants: The iterable of TaskVariantType to include in the learn block.
    :return: A :class:`LearnBlock` with one or more :class:`TaskBlock`s which
        contain the `task_variants` parameter.
    """
    return LearnBlock(split_task_variants(task_variants))


def simple_eval_block(
    task_variants: typing.Iterable[TaskVariantType],
) -> AbstractEvalBlock[TaskVariantType]:
    """
    Constucts an eval block with the task variants passed in. Task blocks are divided as needed.

    :param task_variants: The iterable of TaskVariantType to include in the eval block.
    :return: A :class:`EvalBlock` with one or more :class:`TaskBlock`s which
        contain the `task_variants` parameter.
    """
    return EvalBlock(split_task_variants(task_variants))


Observation = typing.TypeVar("Observation")
Action = typing.TypeVar("Action")
Reward = float
Done = bool
NextObservation = Observation

Transition = typing.Tuple[Observation, Action, Reward, Done, NextObservation]
"""
A tuple with data containing data from a single step in an MDP.
The last item of the tuple is the observation resulting from applying the action
to the observation (i.e. Next observation).
"""

ActionFn = typing.Callable[
    [typing.List[typing.Optional[Observation]]], typing.List[typing.Optional[Action]]
]
"""
A function that takes a list of Observations and returns a list of Actions, one
for each observation.
"""

AbstractRLTaskVariant = AbstractTaskVariant[
    ActionFn, typing.Iterable[Transition], gym.Env
]
"""
An AbstractRLTaskVariant is an TaskVariant that takes an ActionFn as input
and produces an Iterable[Transition]. It also  returns a :class:`gym.Env` as the
Information.
"""


class EpisodicTaskVariant(AbstractRLTaskVariant):
    """
    Represents a TaskVariant that consists of a set number of episodes in a
    :class:`gym.Env`.

    This is a concrete subclass of the :class:`AbstractRLTaskVariant`,
    that takes an :type:`ActionFn` and returns an iterable of :type:`Transition`.
    """

    def __init__(
        self,
        task_cls: typing.Type[gym.Env],
        *,
        num_episodes: int,
        rng_seed: int,
        num_envs: typing.Optional[int] = None,
        params: typing.Optional[typing.Dict] = None,
        task_label: typing.Optional[str] = None,
        variant_label: typing.Optional[str] = "Default",
    ) -> None:
        if num_envs is None:
            num_envs = 1
        if params is None:
            params = {}
        if task_label is None:
            task_label = task_cls.__name__
        assert num_envs > 0

        self._task_cls = task_cls
        self._params = params
        self._num_episodes = num_episodes
        self._num_envs = num_envs
        self._env = None
        self._task_label = task_label
        self._variant_label = variant_label
        self.rng_seed = rng_seed
        self.data_logger = None
        self.logger_info = None
        self.render = False

    def set_render(self, render):
        self.render = render

    @property
    def total_episodes(self):
        return self._num_episodes

    @property
    def task_label(self) -> str:
        return self._task_label

    @property
    def variant_label(self) -> str:
        return self._variant_label

    def validate(self) -> None:
        return validate_params(self._task_cls, list(self._params.keys()))

    def _make_env(self) -> gym.Env:
        """
        Initializes the gym environment object and wraps in the L2MEnv to log rewards
        """
        if self.data_logger is not None:
            return L2LoggerEnv(
                self._task_cls(**self._params), self.data_logger, self.logger_info
            )
        else:
            # FIXME: remove this after #31. this is to support getting spaces without setting l2logger info
            return self._task_cls(**self._params)

    def set_logger_info(
        self, data_logger, block_num: int, is_learning_allowed: bool, exp_num: int
    ):
        self.data_logger = data_logger
        self.logger_info = {
            "block_num": block_num,
            "block_type": "train" if is_learning_allowed else "test",
            "task_params": self._params,
            "task_name": self.task_label + "_" + self.variant_label,
            "worker_id": "worker-default",
            "exp_num": exp_num,
        }

    def info(self) -> gym.Env:
        if self._env is None:
            vector_env_cls = gym.vector.AsyncVectorEnv
            if self._num_envs == 1:
                vector_env_cls = gym.vector.SyncVectorEnv
            self._env = vector_env_cls([self._make_env for _ in range(self._num_envs)])
        return self._env

    def generate(self, action_fn: ActionFn) -> typing.Iterable[Transition]:
        env = self.info()
        env.seed(self.rng_seed)
        num_episodes_finished = 0

        # data to keep track of which observations to mask out (set to None)
        episode_ids = list(range(self._num_envs))
        next_episode_id = episode_ids[-1] + 1

        observations = env.reset()
        while num_episodes_finished < self._num_episodes:
            # mask out any environments that have episode id above max episodes
            mask = [ep_id >= self._num_episodes for ep_id in episode_ids]

            # replace masked environment observations with None
            masked_observations = _where(mask, None, observations)

            # query for the actions
            actions = action_fn(masked_observations)

            # replace masked environment actions with random action
            unmasked_actions = _where(mask, env.single_action_space.sample(), actions)

            # step in the VectorEnv
            next_observations, rewards, dones, infos = env.step(unmasked_actions)
            if self.render:
                env.envs[0].render()
            # yield all the non masked transitions
            for i in range(self._num_envs):
                if not mask[i]:
                    yield (
                        observations[i],
                        actions[i],
                        rewards[i],
                        dones[i],
                        infos[i]["terminal_observation"]
                        if dones[i]
                        else next_observations[i],
                    )

                # increment episode ids if episode ended
                if dones[i]:
                    episode_ids[i] = next_episode_id
                    next_episode_id += 1

            observations = next_observations
            num_episodes_finished += sum(dones)
        self._env.close()
        self._env = None


def _where(
    condition: typing.List[bool], replace_value: typing.Any, original_list: typing.List
) -> typing.List:
    """
    Replaces elements in `original_list[i]` with `replace_value` where the `condition[i]`
    is True.

    :param condition: List of booleans indicating where to put replace_value
    :param replace_value: The value to insert into the list
    :param original_list: The list of values to modify
    :return: A new list with replace_value inserted where condition elements are True
    """
    return [
        replace_value if condition[i] else original_list[i]
        for i in range(len(condition))
    ]


def summarize_curriculum(
    curriculum: AbstractCurriculum[AbstractTaskVariant],
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
                    f"{maybe_plural(task_variant.total_episodes, 'episode')}."
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
    curriculum: AbstractCurriculum[AbstractTaskVariant],
):
    """
    Helper function to do a partial check that task variants are specified
    correctly in all of the blocks of the `curriculum`.

    Uses :meth:`AbstractTaskVariant.validate()` to check task variants.

    Raises a :class:`ValueError` if an invalid parameter is detected.
    Raises a :class:`ValueError` if the curriculum contains multiple observation or action spaces.
    Raises a :class:`ValueError` if any task block contains multiple tasks.
    Raises a :class:`ValueError` if the curriculum, or any block, or any task block is empty.

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
                except Exception as e:
                    raise ValueError(
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
                env = task_variant.info()
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
                        raise ValueError(
                            "All environments in a curriculum must use the same observation and action spaces."
                        )

            # Check that task blocks only contain one task
            if len(task_labels) > 1:
                raise ValueError(
                    f"Block #{i_block}, task block #{i_task_block} had more than 1"
                    f" task label found across all task variants: {task_labels}"
                )

            # Check that no empty blocks are included
            if empty_task:
                raise ValueError(
                    f"Block #{i_block}, task block #{i_task_block} is empty."
                )
        if empty_block:
            raise ValueError(f"Block #{i_block} is empty.")
    if empty_curriculum:
        raise ValueError(f"This curriculum is empty.")


def validate_params(fn: typing.Callable, param_names: typing.List[str]) -> None:
    """
    Determines whether there are missing or invalid names in `param_names` to pass
    to the function `fn`.

    NOTE: if `fn` has any **kwargs, then all arguments are valid and this method
    won't be able to verify anything.

    :param fn: The callable that will accept the parameters.
    :param param_names: The names of the parameters to check.

    :raises: a ValueError if any of `param_names` are not found in the signature, and there are no **kwargs
    :raises: a ValueError if any of the parameters without defaults in `fn` are not present in `param_names`
    :raises: a ValueError if any `*args` are found
    :raises: a ValueError if any positional only arguments are found (i.e. using /)
    """

    fn_signature = inspect.signature(fn)

    kwarg_found = False
    expected_fn_names = []
    for param_name, param in fn_signature.parameters.items():
        if param.kind == param.VAR_POSITIONAL:
            raise ValueError(f"*args not allowed. Found {param_name} in {fn_signature}")
        if param.kind == param.POSITIONAL_ONLY:
            raise ValueError(
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
        raise ValueError(f"Parameters not accepted: {invalid_params} in {fn_signature}")

    # NOTE: this is checking for parameters that don't have a default specified
    missing_params = []
    for name in expected_fn_names:
        if name not in param_names:
            missing_params.append(name)
    if len(missing_params) > 0:
        raise ValueError(f"Missing parameters: {missing_params} in {fn_signature}")

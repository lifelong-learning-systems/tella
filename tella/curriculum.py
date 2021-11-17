import typing
import gym
import abc
import inspect


class TaskBlock:
    """
    Represents a block of episodes that use the same task class & task params.

    This is the basic building block of a lifelong learning curriculum.

    NOTE: does not instantiate the gym.Env object until :meth:`TaskBlock.task()` is called
    """

    def __init__(
        self,
        task_cls: typing.Type[gym.Env],
        *,
        num_episodes: int,
        params: typing.Optional[typing.Dict] = None,
    ):
        if params is None:
            params = {}
        self.task_cls = task_cls
        self.params = params
        self.num_episodes = num_episodes

    def task(self) -> gym.Env:
        """
        Construct the gym environment with the associated parameters
        :return: :class:`gym.Env`
        """
        return self.task_cls(**self.params)

    def get_invalid_params(self) -> typing.List[str]:
        """
        Determines whether any of the parameters do not match the signature
        of the class constructor using the `inspect` package.

        :return: List of parameter names that do not match with the signature
            returned from inspect
        """
        invalid_params = []
        task_signature = inspect.signature(self.task_cls)
        for name, _value in self.params.items():
            if name not in task_signature.parameters:
                invalid_params.append(name)
        return invalid_params

    def __repr__(self) -> str:
        return f"TaskBlock({self.task_cls.__name__}, num_episodes={self.num_episodes}, params={self.params})"


class Curriculum(abc.ABC):
    """
    Represents a curriculum for lifelong/continual RL.

    At a high level a curriculum is comprised of a sequence of tasks (i.e. :class:`gym.Env`)
    that the agent will learn from.

    This class organizes the curriculum into two parts:
    1. Learning blocks, which are a sequence of 1+ :class:`TaskBlock`
    2. Evaluation block, which is a 1= :class:`TaskBlock`
    """

    observation_space: gym.Space
    action_space: gym.Space

    @abc.abstractmethod
    def learning_blocks(self) -> typing.Iterable[typing.Iterable[TaskBlock]]:
        pass

    @abc.abstractmethod
    def eval_block(self) -> typing.Iterable[TaskBlock]:
        pass

    def validate(self):
        """
        Helper function to do a partial check that task parameters are specified
        correctly.

        This loops through all learning blocks and the eval block to check
        if the parameters are specified correctly.

        See :class:`TaskBlock.get_invalid_params()` for how this checks for
        incorrect parameters.

        :return: None
        """
        for i_block, learning_block in enumerate(self.learning_blocks()):
            for i_task, task_block in enumerate(learning_block):
                invalids = task_block.get_invalid_params()
                if len(invalids) > 0:
                    lines = [
                        f"Incompatible task parameters detected in learning block #{i_block}, task block #{i_task}. {task_block}",
                        f"Invalid parameters: {invalids}",
                        f"Task class signature: {inspect.signature(task_block.task_cls)}",
                    ]
                    raise ValueError("\n".join(lines))

        for task_block in self.eval_block():
            invalids = task_block.get_invalid_params()
            if len(invalids) > 0:
                lines = [
                    f"Incompatible task parameters detected in learning block #{i_block}, task block #{i_task}. {task_block}",
                    f"Invalid parameters: {invalids}",
                    f"Task class signature: {inspect.signature(task_block.task_cls)}",
                ]
                raise ValueError("\n".join(lines))

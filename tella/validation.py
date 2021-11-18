import typing
import inspect
from .curriculum import Curriculum


def validate_curriculum(curriculum: Curriculum):
    """
    Helper function to do a partial check experiences are specified
    correctly in all of the blocks of the `curriculum`.

    Uses :meth:`Experience.validate()` to check experiences.

    Raises a :class:`ValueError` if an invalid parameter is detected.

    :return: None
    """
    for i_block, block in enumerate(curriculum.blocks()):
        for i_experience, experience in enumerate(block.experiences()):
            try:
                experience.validate()
            except Exception as e:
                raise ValueError(
                    f"Invalid experience at block #{i_block}, experience #{i_experience}.",
                    e,
                )


def validate_params(fn: typing.Any, param_names: typing.List[str]) -> None:
    """
    Determines whether any of the parameters for the `task_experience` do not
    match the signature of the `task_class` constructor using the `inspect` package.

    NOTE: this is not guaranteed to be correct, due to unknown behavior
        with **kwargs. This will only catch typos in named parameters

    :param fn: The callable that will accept the parameters.
    :param param_names: The names of the parameters to check.

    Raises a ValueError if any of the parameters are incorrectly named.
    """
    if len(param_names) == 0:
        return

    invalid_params = []
    fn_signature = inspect.signature(fn)
    for name in param_names:
        if name not in fn_signature.parameters:
            invalid_params.append(name)
    if len(invalid_params) > 0:
        raise ValueError(
            f"Invalid parameters: {invalid_params}",
            f"Function Signature {fn_signature}",
        )

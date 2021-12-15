import typing
import inspect
import warnings
from .curriculum import AbstractCurriculum, AbstractTaskVariant


def validate_curriculum(curriculum: AbstractCurriculum[AbstractTaskVariant]):
    """
    Helper function to do a partial check that task variants are specified
    correctly in all of the blocks of the `curriculum`.

    Uses :meth:`AbstractTaskVariant.validate()` to check task variants.

    Raises a :class:`ValueError` if an invalid parameter is detected.

    :return: None
    """
    for i_block, block in enumerate(curriculum.learn_blocks_and_eval_blocks()):
        for i_task_block, task_block in enumerate(block.task_blocks()):
            task_labels = set()
            num_task_variants = 0
            variant_labels = set()
            for i_task_variant, task_variant in enumerate(task_block.task_variants()):
                task_labels.add(task_variant.task_label())
                variant_labels.add(task_variant.variant_label())
                num_task_variants += 1
                try:
                    task_variant.validate()
                except Exception as e:
                    raise ValueError(
                        f"Invalid task variant at block #{i_block}, task block #{i_task_block}, task variant #{i_task_variant}.",
                        e,
                    )
            if len(task_labels) != 1:
                raise ValueError(
                    f"Block #{i_block}, task block #{i_task_block} had more than 1 task label found across all task variants:"
                    f"{task_labels}"
                )
            if len(variant_labels) != num_task_variants:
                warnings.warn(
                    "Multiple task variants shared the same variant label."
                    "Consider combining these task variants."
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

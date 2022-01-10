import pytest
from tella.curriculum import curriculum_registry, validate_curriculum


def test_contents():
    num_expected = 2
    try:
        import gym_minigrid
    except ImportError:
        pass
    else:
        num_expected += 3
    assert len(curriculum_registry) == num_expected


@pytest.mark.parametrize("curriculum_name", list(curriculum_registry.keys()))
def test_validate(curriculum_name):
    curriculum_cls = curriculum_registry[curriculum_name]
    curriculum = curriculum_cls(0)
    validate_curriculum(curriculum)

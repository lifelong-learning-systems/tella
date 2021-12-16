from tella.curriculum import curriculum_registry


def test_contents():
    num_expected = 1
    try:
        import gym_minigrid
    except ImportError:
        pass
    else:
        num_expected += 1
    assert len(curriculum_registry) == num_expected

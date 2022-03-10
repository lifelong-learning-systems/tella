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

import gym
import numpy as np


# fmt: off
ATARI_ACTIONS = [
    "NOOP",
    "FIRE",
    "UP",
    "RIGHT",
    "LEFT",
    "DOWN",
    "UPRIGHT",
    "UPLEFT",
    "DOWNRIGHT",
    "DOWNLEFT",
    "UPFIRE",
    "RIGHTFIRE",
    "LEFTFIRE",
    "DOWNFIRE",
    "UPRIGHTFIRE",
    "UPLEFTFIRE",
    "DOWNRIGHTFIRE",
    "DOWNLEFTFIRE",
]
ATARI_ACTION_INDEX = {action: index for index, action in enumerate(ATARI_ACTIONS)}
ATARI_ACTION_REPLACEMENT = {
    "NOOP": [],
    "FIRE": ["NOOP"],
    "UP": ["NOOP"],
    "RIGHT": ["NOOP"],
    "LEFT": ["NOOP"],
    "DOWN": ["NOOP"],
    "UPRIGHT": ["UP", "RIGHT", "NOOP"],
    "UPLEFT": ["UP", "LEFT", "NOOP"],
    "DOWNRIGHT": ["DOWN", "RIGHT", "NOOP"],
    "DOWNLEFT": ["DOWN", "LEFT", "NOOP"],
    "UPFIRE": ["UP", "FIRE", "NOOP"],
    "RIGHTFIRE": ["RIGHT", "FIRE", "NOOP"],
    "LEFTFIRE": ["LEFT", "FIRE", "NOOP"],
    "DOWNFIRE": ["DOWN", "FIRE", "NOOP"],
    "UPRIGHTFIRE": ["UPRIGHT", "UPFIRE", "RIGHTFIRE", "UP", "RIGHT", "FIRE", "NOOP"],
    "UPLEFTFIRE": ["UPLEFT", "UPFIRE", "LEFTFIRE", "UP", "LEFT", "FIRE", "NOOP"],
    "DOWNRIGHTFIRE": ["DOWNRIGHT", "DOWNFIRE", "RIGHTFIRE", "DOWN", "RIGHT", "FIRE", "NOOP"],
    "DOWNLEFTFIRE": ["DOWNLEFT", "DOWNFIRE", "LEFTFIRE", "DOWN", "LEFT", "FIRE", "NOOP"],
}
# fmt: on


class AtariStandardizedActionSpaceWrapper(gym.ActionWrapper):
    """Make all 18 actions available for each Atari environment."""

    def __init__(self, env: gym.Env) -> None:
        assert isinstance(env.action_space, gym.spaces.Discrete)
        super().__init__(env)

        # Get the expected actions from the Atari env
        expected_actions = env.unwrapped.get_action_meanings()
        expected_action_index = {
            action: index for index, action in enumerate(expected_actions)
        }
        # Map all other Atari actions to one of the expected actions in this env
        self.action_map = {}
        for action in ATARI_ACTIONS:
            if action in expected_actions:
                self.action_map[ATARI_ACTION_INDEX[action]] = expected_action_index[
                    action
                ]
            else:
                for replacement_action in ATARI_ACTION_REPLACEMENT[action]:
                    if replacement_action in expected_actions:
                        self.action_map[
                            ATARI_ACTION_INDEX[action]
                        ] = expected_action_index[replacement_action]
                        break
                else:  # iff none of the replacement actions were in expected_actions
                    raise KeyError(
                        f"Env {env} could not be mapped to standard Atari action space!"
                    )

        self.action_space = gym.spaces.Discrete(len(ATARI_ACTIONS))

    def action(self, act):
        if act not in self.action_map:
            raise gym.error.InvalidAction()
        return self.action_map[act]


class AtariStandardizedObservationSpaceWrapper(gym.ObservationWrapper):
    """Make all Atari games return the full screen height."""

    def __init__(self, env: gym.Env) -> None:
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert len(env.observation_space.shape) == 3
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 255, (250, 160, 3), dtype=np.uint8)

    def observation(self, observation):
        missing_height = 250 - observation.shape[0]
        if missing_height:
            observation = np.concatenate(
                [observation, np.zeros((missing_height, 160, 3), dtype=np.uint8)],
                axis=0,
            )
        return observation


class _AtariEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env = AtariStandardizedActionSpaceWrapper(self.env)
        self.env = AtariStandardizedObservationSpaceWrapper(self.env)


class AirRaid(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/AirRaid-v5"))


class Alien(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Alien-v5"))


class Amidar(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Amidar-v5"))


class Assault(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Assault-v5"))


class Asterix(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Asterix-v5"))


class Asteroids(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Asteroids-v5"))


class Atlantis(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Atlantis-v5"))


class BankHeist(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/BankHeist-v5"))


class BattleZone(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/BattleZone-v5"))


class BeamRider(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/BeamRider-v5"))


class Berzerk(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Berzerk-v5"))


class Bowling(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Bowling-v5"))


class Boxing(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Boxing-v5"))


class Breakout(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Breakout-v5"))


class Carnival(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Carnival-v5"))


class Centipede(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Centipede-v5"))


class ChopperCommand(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/ChopperCommand-v5"))


class CrazyClimber(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/CrazyClimber-v5"))


class DemonAttack(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/DemonAttack-v5"))


class DoubleDunk(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/DoubleDunk-v5"))


class ElevatorAction(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/ElevatorAction-v5"))


class Enduro(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Enduro-v5"))


class FishingDerby(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/FishingDerby-v5"))


class Freeway(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Freeway-v5"))


class Frostbite(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Frostbite-v5"))


class Gopher(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Gopher-v5"))


class Gravitar(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Gravitar-v5"))


class IceHockey(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/IceHockey-v5"))


class Jamesbond(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Jamesbond-v5"))


class JourneyEscape(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/JourneyEscape-v5"))


class Kangaroo(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Kangaroo-v5"))


class Krull(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Krull-v5"))


class KungFuMaster(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/KungFuMaster-v5"))


class MontezumaRevenge(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/MontezumaRevenge-v5"))


class MsPacman(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/MsPacman-v5"))


class NameThisGame(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/NameThisGame-v5"))


class Phoenix(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Phoenix-v5"))


class Pitfall(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Pitfall-v5"))


class Pong(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Pong-v5"))


class Pooyan(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Pooyan-v5"))


class PrivateEye(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/PrivateEye-v5"))


class Qbert(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Qbert-v5"))


class Riverraid(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Riverraid-v5"))


class RoadRunner(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/RoadRunner-v5"))


class Robotank(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Robotank-v5"))


class Seaquest(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Seaquest-v5"))


class Skiing(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Skiing-v5"))


class Solaris(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Solaris-v5"))


class SpaceInvaders(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/SpaceInvaders-v5"))


class StarGunner(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/StarGunner-v5"))


class Tennis(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Tennis-v5"))


class TimePilot(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/TimePilot-v5"))


class Tutankham(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Tutankham-v5"))


class UpNDown(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/UpNDown-v5"))


class Venture(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Venture-v5"))


class VideoPinball(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/VideoPinball-v5"))


class WizardOfWor(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/WizardOfWor-v5"))


class YarsRevenge(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/YarsRevenge-v5"))


class Zaxxon(_AtariEnv):
    def __init__(self):
        super().__init__(gym.make("ALE/Zaxxon-v5"))


ATARI_TASKS = {
    "AirRaid": AirRaid,
    "Alien": Alien,
    "Amidar": Amidar,
    "Assault": Assault,
    "Asterix": Asterix,
    "Asteroids": Asteroids,
    "Atlantis": Atlantis,
    "BankHeist": BankHeist,
    "BattleZone": BattleZone,
    "BeamRider": BeamRider,
    "Berzerk": Berzerk,
    "Bowling": Bowling,
    "Boxing": Boxing,
    "Breakout": Breakout,
    "Carnival": Carnival,
    "Centipede": Centipede,
    "ChopperCommand": ChopperCommand,
    "CrazyClimber": CrazyClimber,
    "DemonAttack": DemonAttack,
    "DoubleDunk": DoubleDunk,
    "ElevatorAction": ElevatorAction,
    "Enduro": Enduro,
    "FishingDerby": FishingDerby,
    "Freeway": Freeway,
    "Frostbite": Frostbite,
    "Gopher": Gopher,
    "Gravitar": Gravitar,
    "IceHockey": IceHockey,
    "Jamesbond": Jamesbond,
    "JourneyEscape": JourneyEscape,
    "Kangaroo": Kangaroo,
    "Krull": Krull,
    "KungFuMaster": KungFuMaster,
    "MontezumaRevenge": MontezumaRevenge,
    "MsPacman": MsPacman,
    "NameThisGame": NameThisGame,
    "Phoenix": Phoenix,
    "Pitfall": Pitfall,
    "Pong": Pong,
    "Pooyan": Pooyan,
    "PrivateEye": PrivateEye,
    "Qbert": Qbert,
    "Riverraid": Riverraid,
    "RoadRunner": RoadRunner,
    "Robotank": Robotank,
    "Seaquest": Seaquest,
    "Skiing": Skiing,
    "Solaris": Solaris,
    "SpaceInvaders": SpaceInvaders,
    "StarGunner": StarGunner,
    "Tennis": Tennis,
    "TimePilot": TimePilot,
    "Tutankham": Tutankham,
    "UpNDown": UpNDown,
    "Venture": Venture,
    "VideoPinball": VideoPinball,
    "WizardOfWor": WizardOfWor,
    "YarsRevenge": YarsRevenge,
    "Zaxxon": Zaxxon,
}

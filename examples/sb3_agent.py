import typing
import stable_baselines3
from tella.agents.continual_rl_agent import ContinualRLAgent
from tella.agents.metrics.rl import default_metrics, RLMetricAccumulator, MDPTransition

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from tella.experiences.rl import Observation, Action
from rl_logging_agent import LoggingAgent

sb3agent = typing.Union[OnPolicyAlgorithm, OffPolicyAlgorithm]

class ContinualRLSB3Agent(LoggingAgent):
    """
    The base class for wrapping a stable baselines model in tella.
    This model will take in a :class: OnPolicyAgent or :class: OffPolicyAgent
    and implement step_observe and step_transition
    """
    def __init__(
        self,
        sb3_model : sb3agent,
        num_envs: int,
    ):
        self.model = sb3_model
        observation_space = sb3_model.env.observation_space
        action_space = sb3_model.env.action_space
        super().__init__(observation_space, action_space, num_envs)

    def step_observe(self, observations: typing.List[typing.Optional[Observation]]) -> typing.List[typing.Optional[Action]]:
        actions, _ = self.model.predict(observations)
        return actions
    
    def step_transition(self, transition: MDPTransition) -> bool:
        return True


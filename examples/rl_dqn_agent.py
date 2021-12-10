import collections
import logging
import random
import typing

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tella
from tella.agents.continual_rl_agent import (
    ContinualRLAgent,
    Observation,
    Action,
    Metrics,
)
from tella.agents.metrics.rl import RLMetricAccumulator
from tella.curriculum import *
from tella.curriculum.rl_task_variant import (
    StepData,
    AbstractRLTaskVariant,
    EpisodicTaskVariant,
)
from tella.curriculum.builders import simple_eval_block, simple_learn_block
from tella.run import run


logger = logging.getLogger(__name__)


# Adapting code from minimalRL repo: https://github.com/seungeunrho/minimalRL/blob/master/dqn.py
# To create a simple DQN agent using the TELLA API


# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32


class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return (
            torch.tensor(s_lst, dtype=torch.float),
            torch.tensor(a_lst),
            torch.tensor(r_lst),
            torch.tensor(s_prime_lst, dtype=torch.float),
            torch.tensor(done_mask_lst),
        )

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()


def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# minimalRL writes DQN in a script, so translation here requires moving components into the agent class
class MinimalRlDqnAgent(ContinualRLAgent):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        num_envs: int,
        metric: typing.Optional[RLMetricAccumulator] = None,
    ) -> None:
        super(MinimalRlDqnAgent, self).__init__(
            observation_space, action_space, num_envs, metric
        )
        logger.info(
            f"Constructed with observation_space={observation_space} "
            f"action_space={action_space} num_envs={num_envs}"
        )

        self.q = Qnet()
        self.q_target = Qnet()
        self.q_target.load_state_dict(self.q.state_dict())
        self.memory = ReplayBuffer()
        self.optimizer = optim.Adam(self.q.parameters(), lr=learning_rate)
        self.training = None
        self.epsilon = 0.01
        self.num_eps_done = 0
        self.q_target_interval = 20

    def block_start(self, is_learning_allowed: bool) -> None:
        super().block_start(is_learning_allowed)
        if is_learning_allowed:
            logger.info("About to start a new learning block")
            self.training = True
        else:
            logger.info("About to start a new evaluation block")
            self.training = False

    def task_start(self, task_name: typing.Optional[str]) -> None:
        logger.info(
            f"\tAbout to start interacting with a new task. task_name={task_name}"
        )

    def consume_task_variant(self, task_variant: AbstractRLTaskVariant) -> Metrics:
        logger.info("\tConsuming task variant")
        return super().consume_task_variant(task_variant)

    def step_observe(
        self, observations: typing.List[typing.Optional[Observation]]
    ) -> typing.List[typing.Optional[Action]]:
        logger.debug(f"\t\t\tReturn {len(observations)} actions")
        return [
            None
            if obs is None
            else self.q.sample_action(
                torch.from_numpy(obs).float(), self.epsilon if self.training else 0.0
            )
            for obs in observations
        ]

    def step_transition(self, step: StepData):
        s, a, r, done, s_prime = step
        self.memory.put((s, a, r / 100.0, s_prime, 0.0 if done else 1.0))
        logger.debug(f"\t\t\tReceived transition done={done}")

        # Handle end-of-episode matters: training, logging, and annealing
        if done:
            self.num_eps_done += 1

            logger.info(
                f"\t\t"
                f"n_episode: {self.num_eps_done}, "
                f"score: {self.metric.calculate()['MeanEpisodeReward']:.1f}, "
                f"n_buffer: {self.memory.size()}, "
                f"eps: {self.epsilon*100:.1f}%"
            )

            if self.memory.size() > 100:  # was 2000 in minimalRL repo
                logger.info(f"\t\tTraining Q network")
                train(self.q, self.q_target, self.memory, self.optimizer)

            if self.num_eps_done % self.q_target_interval == 0:
                logger.info(f"\t\tUpdating target Q network")
                self.q_target.load_state_dict(self.q.state_dict())

            self.epsilon = max(
                0.01, 0.08 - 0.01 * (self.num_eps_done / 200)
            )  # Linear annealing from 8% to 1%

    def task_end(self, task_name: typing.Optional[str]) -> None:
        logger.info(f"\tDone interacting with task. task_name={task_name}")

    def block_end(self, is_learning_allowed: bool) -> None:
        if is_learning_allowed:
            logger.info("Done with learning block")
        else:
            logger.info("Done with evaluation block")


class ExampleCurriculum(AbstractCurriculum[AbstractRLTaskVariant]):
    def learn_blocks_and_eval_blocks(
        self,
    ) -> typing.Iterable[
        typing.Union[
            "AbstractLearnBlock[TaskVariantType]", "AbstractEvalBlock[TaskVariantType]"
        ]
    ]:
        yield simple_learn_block(
            [EpisodicTaskVariant(lambda: gym.make("CartPole-v1"), num_episodes=1_000)]
        )
        yield simple_eval_block(
            [EpisodicTaskVariant(lambda: gym.make("CartPole-v1"), num_episodes=100)]
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    tella.rl_cli(MinimalRlDqnAgent, ExampleCurriculum)

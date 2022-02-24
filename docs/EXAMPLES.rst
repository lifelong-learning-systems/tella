Example agents
====================
Several example agents are included with tella to demonstrate use of the agent API.

Note that each is a subclass of :class:`tella.agents.ContinualRLAgent`,
and that each includes a block that enables use of the tella CLI::

    if __name__ == "__main__":
        tella.rl_cli(<ThisAgentClass>)

Minimal RL Agent
-----------------
This simple agent chooses random actions and does not use the transition data.
Handling of curriculum events is inherited from
:class:`ContinualRLAgent <tella.agents.ContinualRLAgent>`, and
only :meth:`choose_actions() <tella.agents.ContinualRLAgent.choose_actions>`
and :meth:`receive_transitions() <tella.agents.ContinualRLAgent.receive_transitions>`
are be defined.::

    class MinimalRandomAgent(tella.ContinualRLAgent):
        def choose_actions(self, observations):
            """Loop over the environments' observations and select action"""
            return [
                None if obs is None else self.action_space.sample() for obs in observations
            ]

        def receive_transitions(self, transitions):
            """Do nothing here since we are not learning"""
            pass

Run this agent by::

    python examples/rl_minimal_agent.py --curriculum SimpleCartPole


Logging Agent
---------------
This logs every event the agent has access to. For example::

    class LoggingAgent(tella.ContinualRLAgent):
        ...
        def task_variant_start(
            self,
            task_name: typing.Optional[str],
            variant_name: typing.Optional[str],
        ) -> None:
            logger.info(
                f"\tAbout to start interacting with a new task variant. "
                f"task_name={task_name} variant_name={variant_name}"
            )
        ...

Run this agent by::

    python examples/rl_logging_agent.py --curriculum SimpleCartPole


DQN RL Agent
--------------
This uses code from the
`minimalRL <https://github.com/seungeunrho/minimalRL/>`_
package to create an agent that learns.

This agent requires `PyTorch <https://pytorch.org/>`_: ``pip install torch``

Run this agent by::

    python examples/rl_dqn_agent.py --curriculum SimpleCartPole

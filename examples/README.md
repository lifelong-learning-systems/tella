Examples for tella
====================
To better understand the API, examples have been implemented.
They all use the tella command line interface.
The curriculum is the only required argument.
The CLI writes out the L2Logger files by default to a `logs` directory.

Minimal RL Agent
-----------------
This agent chooses random actions and does not use the transition data.
It is the simplest agent that can be implemented.

Running:`python rl_minimal_agent.py --curriculum SimpleCartPole`


Logging Agent
---------------
This logs every event the agent has access to so that you can see their flow.

Running: `python rl_logging_agent.py --curriculum SimpleCartPole`


DQN RL Agent
--------------
This uses code from the minimalRL package to create an agent that learns.

Installing: `pip install torch`

Running: `python rl_dqn_agent.py --curriculum SimpleCartPole`

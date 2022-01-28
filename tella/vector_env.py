import typing
import sys
from gym.vector.async_vector_env import AsyncVectorEnv, AsyncState
import numpy as np
import multiprocessing as mp
import sys
from gym.vector.vector_env import VectorEnv
from gym.vector.utils import create_empty_array, clear_mpi_env_vars


def _env_settable_worker(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is None
    env = env_fn()
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == "step":
                observation, reward, done, info = env.step(data)
                if done:
                    info["terminal_observation"] = observation
                    observation = env.reset()
                pipe.send(((observation, reward, done, info), True))
            elif command == "reset":
                observation = env.reset()
                pipe.send((observation, True))
            elif command == "set_env":
                # NOTE: this command is added for TELLA
                env = data()
                pipe.send((None, True))
            elif command == "render":
                # NOTE: this command is added for TELLA
                env.render()
                pipe.send((None, True))
            elif command == "seed":
                env.seed(data)
                pipe.send((None, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_check_observation_space":
                pipe.send((data == env.observation_space, True))
            else:
                raise RuntimeError(
                    "Received unknown command `{0}`. Must "
                    "be one of {`reset`, `step`, `seed`, `close`, "
                    "`_check_observation_space`}.".format(command)
                )
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()


def none_factory():
    return None


class WorkerPool:
    def __init__(
        self,
        num_processes: int,
        context: typing.Optional[str] = None,
        target=_env_settable_worker,
        daemon=True,
    ) -> None:
        self.num_processes = num_processes
        self.ctx = mp.get_context(context)
        self.pipes = []
        self.processes = []
        self.error_queue = self.ctx.Queue()

        with clear_mpi_env_vars():
            for i in range(num_processes):
                parent_pipe, child_pipe = self.ctx.Pipe()
                process = self.ctx.Process(
                    target=target,
                    name=f"Worker<{type(self).__name__}>-{i}",
                    args=(
                        i,
                        none_factory,
                        child_pipe,
                        parent_pipe,
                        None,
                        self.error_queue,
                    ),
                )

                self.pipes.append(parent_pipe)
                self.processes.append(process)

                process.daemon = daemon
                process.start()
                child_pipe.close()

    def close(self):
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for pipe in self.pipes:
            if pipe is not None:
                pipe.close()
        for process in self.processes:
            process.join()


class PooledVectorEnv(AsyncVectorEnv):
    def __init__(
        self, env_fns, worker_pool: typing.Optional[WorkerPool] = None, **kwargs
    ):
        self.worker_pool = worker_pool

        if worker_pool is None:
            super().__init__(
                env_fns,
                worker=_env_settable_worker,
                shared_memory=False,
                copy=True,
                **kwargs,
            )
        else:
            env_fn = env_fns[0]
            dummy_env = env_fn()
            observation_space = dummy_env.observation_space
            action_space = dummy_env.action_space
            dummy_env.close()
            del dummy_env

            VectorEnv.__init__(
                self,
                num_envs=worker_pool.num_processes,
                observation_space=observation_space,
                action_space=action_space,
            )
            self.shared_memory = False
            self.copy = True
            self.observations = create_empty_array(
                self.single_observation_space, n=self.num_envs, fn=np.zeros
            )
            self.parent_pipes = worker_pool.pipes
            self.processes = worker_pool.processes
            self.error_queue = worker_pool.error_queue
            self._state = AsyncState.DEFAULT
            self.set_env(env_fn)
            self._check_observation_spaces()

    def set_env(self, env_fn):
        for pipe in self.parent_pipes:
            pipe.send(("set_env", env_fn))
        _, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)

    def render(self):
        self.parent_pipes[0].send(("render", None))
        _, success = self.parent_pipes[0].recv()
        self._raise_if_errors([success])

    def close_extras(self, **kwargs) -> None:
        if self.worker_pool is not None:
            self.parent_pipes = []
            self.processes = []
            self.error_queue = None
        return super().close_extras(**kwargs)

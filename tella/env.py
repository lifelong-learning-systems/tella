import typing
import gym


class L2MEnv(gym.Wrapper):
    """
    A wrapper around the gym environment to record the reward at each episode end
    """

    def __init__(self, env: gym.Env, data_logger, logger_info):
        super().__init__(env)
        self.data_logger = data_logger
        self.logger_info = logger_info
        self.logger_info["task_name"] = env.__class__.__name__

    def step(self, action):
        """
        Overwrites the default step function to record the rewards with L2Logger.
        Tracks exp_num via the done signal
        """
        obs, reward, done, info = super().step(action)
        record = self.logger_info
        if done:
            status = "complete"
        else:
            status = "incomplete"
        record.update({"reward": reward, "exp_status": status})
        self.data_logger.log_record(record)
        ##TODO solution to increment episodes
        if done:
            self.logger_info["exp_num"] += 1
        return obs, reward, done, info

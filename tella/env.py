"""
Copyright © 2021 The Johns Hopkins University Applied Physics Laboratory LLC

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

import typing

import gym

from l2logger import l2logger


class L2LoggerEnv(gym.Wrapper):
    """
    A wrapper around the gym environment to record the reward at each episode end
    """

    def __init__(
        self,
        env: gym.Env,
        data_logger: l2logger,
        logger_info: typing.Dict[str, typing.Any],
    ):
        super().__init__(env)
        self.data_logger = data_logger
        self.logger_info = logger_info

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

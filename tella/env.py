from gym import Wrapper

class L2MEnv(Wrapper):
    def __init__(self, env, data_logger, logger_info):
        self.data_logger = data_logger
        self.logger_info = logger_info
        super().__init__(env)
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        record = self.logger_info
        if done:
            status = 'complete'
        else:
            status = 'incomplete'
        record.update({'reward':reward, 'exp_status':status})
        self.data_logger.log_record(record)
        ##TODO solution to increment episodes
        if done:
            self.logger_info['exp_num'] += 1
        return obs, reward, done, info

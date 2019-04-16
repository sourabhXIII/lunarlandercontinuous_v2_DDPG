"""
@author sourabhxiii

"""

import numpy as np

class RandomAgent:
    def __init__(self, env, max_t, episodes):
        self.env = env
        self.max_timesteps = max_t
        self.episodes = episodes
        self.collected_reward = np.zeros((self.episodes, ))
        self.episode_length = np.zeros((self.episodes, ))
    def play(self):
        e = 0
        while e < self.episodes:
            self.env.render()
            self.env.seed(1)
            self.env.reset()
            t = 0
            episode_reward = 0
            while t < self.max_timesteps:
                self.env.render()
                _, r, done, _ = self.env.step(self.env.action_space.sample())
                episode_reward += r
                if done:
                    break
                t += 1
            self.collected_reward[e] = episode_reward
            self.episode_length[e] = t
            e += 1
            print('Episode %d reward collected %d' %(e, episode_reward))
        return self.collected_reward, self.episode_length
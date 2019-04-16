"""
@author sourabhxiii

"""

import numpy as np
import matplotlib.pyplot as plt


class PlotReward:
    """
    Plots the per episode reward and episode lengths.
    """
    def __init__(self, reward_arr, episode_lengths, env_name):
        self.reward_arr = reward_arr
        self.episode_lengths = episode_lengths
        self.env_name = env_name

    def plot(self):
        """
        Plots the per episode reward and episode lengths.
        """
        plt.figure(1)
        plt.subplot(211)
        plt.plot(self.reward_arr)
        plt.ylabel('Rewards')
        plt.xlabel('Episodes')
        if self.episode_lengths is not None:
            plt.subplot(212)
            plt.plot(self.episode_lengths)
            plt.ylabel('Length')
            plt.xlabel('Episodes')

        plt.suptitle("Plots of %s" % self.env_name)

        plt.show()
"""
@author sourabhxiii

To install pybox2d in anaconda on windows follow this:
https://github.com/pybox2d/pybox2d/blob/master/INSTALL.md

Space invaders:
https://stackoverflow.com/questions/42605769/openai-gym-atari-on-windows/46739299
https://github.com/j8lp/atari-py
https://github.com/openai/gym/issues/11
"""

import numpy as np
import gym
import tensorflow as tf
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "./llcv2_utility")

from llcv2_utility.randomagent import RandomAgent
from llcv2_utility.ddpgagent import Agent
from llcv2_utility.plotreward import PlotReward
from llcv2_utility.featuretransformer import FeatureTransformer


SEED = 0

def play_with_a_random_agent(env):
    """
    Play with a random agent. Just to create a baseline.
    """
    ra = RandomAgent(env, 300, 5)
    rewards, episode_lengths = ra.play()
    plotter = PlotReward(rewards, episode_lengths, 'LL-v2-ra')
    plotter.plot()
    

def play_with_a_ddpg_agent(env):
    """
    Play using a smart agent.
    """
    n_episodes=50000
    max_t=500

    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)

    agent = Agent(state_size=env.observation_space.shape[0]
        , axn_size=env.action_space.shape[0], axn_low=env.action_space.low
        , axn_high=env.action_space.high, sess=sess, seed=SEED)
    agent.reset()
    agent.set_session()
    scores = []
    for i_episode in range(1, n_episodes+1):
        env.seed(SEED)
        state = env.reset()
        agent.reset()
        score = 0
        for _ in range(max_t):
            action = agent.act(state)
            rand_action = env.action_space.sample()
            # env.render()
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores), score), end="\n")
    return scores

def main():
    env = gym.make('LunarLanderContinuous-v2')
    env.reset()
    print('observation space:', env.observation_space)
    print('action space:', env.action_space)
    # play_with_a_random_agent(env)
    episode_reward = play_with_a_ddpg_agent(env)
    plotter = PlotReward(episode_reward, None, 'LL-v2-sa')
    plotter.plot()
    env.close()




if __name__ == '__main__':
    main()

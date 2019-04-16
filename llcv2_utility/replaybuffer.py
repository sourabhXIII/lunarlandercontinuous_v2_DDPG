"""
@author sourabhxiii

"""
from collections import deque, namedtuple
import random
import numpy as np

class ReplayBuffer:
    """
    Stores the experience in a buffer.
    """
    def __init__(self, buffer_size, batch_size, seed):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_replay_samples = 0
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.buffer = deque()

    def get_batch(self):
        """
        Returns randomly sampled batch_size replay_samples.
        """
        if self.num_replay_samples < self.batch_size:
            batch = random.sample(self.buffer, self.num_replay_samples)
        else:
            batch = random.sample(self.buffer, self.batch_size)
        return map(np.array, zip(*batch))
    
    def get_size(self):
        """
        Returns the buffer size
        """
        return self.buffer_size

    def get_length(self):
        """
        Returns the current number of replay samples.
        """
        return self.num_replay_samples

    def add_experience(self, state, action, reward, next_state, done):
        """
        Add experience to the replay buffer.
        Experience is a tuple `(state, action, reward, new_state, done)`
        """
        e = self.experience(state, action, reward, next_state, done)
        if self.num_replay_samples < self.buffer_size:
            self.buffer.append(e)
            self.num_replay_samples += 1
        else:
            # make some room, remove the oldest experience.
            self.buffer.popleft()
            self.buffer.append(e)
        return

    def reset(self):
        """
        Purges the replay buffer. Resets the counter.
        """
        self.buffer.clear()
        self.num_replay_samples = 0
        return
"""
@author sourabhxiii

"""

from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
import numpy as np

# Transforms the state into RBF features
class FeatureTransformer:
    """
    Featurizes using RBFSampler with 10000 sample observations of the environment.
    """
    def __init__(self, env, n_components=2):
        # 1. get a few samples to initialize the feature transformer
        # 2. create a standard-normalizer
        # 3. create a featurizer
        # 4. initialize the members

        # 1
        samples = np.array([env.observation_space.sample() for _ in range(10000)])
        # 2
        normalizer = StandardScaler()
        normalizer.fit(samples)
        # 3
        featurizer = FeatureUnion([
        ("rbf1", RBFSampler(gamma=0.5, n_components=n_components)),
        ("rbf2", RBFSampler(gamma=1, n_components=n_components)),
        ("rbf3", RBFSampler(gamma=2, n_components=n_components)),
        ("rbf4", RBFSampler(gamma=5, n_components=n_components)),
        ("rbf5", RBFSampler(gamma=7, n_components=n_components)),
        ])
        # 4
        features = featurizer.fit_transform(normalizer.transform(samples))
        self.dimensions = features.shape[1]
        self.normalizer = normalizer
        self.featurizer = featurizer

    def transform(self, obs):
        # 1. normalize the observation
        # 2. return the featurized observation
        normalized = self.normalizer.transform(obs)
        return self.featurizer.transform(normalized)
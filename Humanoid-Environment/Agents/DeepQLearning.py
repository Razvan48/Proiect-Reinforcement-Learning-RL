import gymnasium as gym
import numpy as np



# TODO: reteaua neuronala pentru Deep Q Learning (convolutie?)

class DeepQLearning:



    def __init__(self):
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.1

        self.experienceReplaySamples = []



    def generateExperienceReplaySamples(self, numSamples : int, env : gym.Env):
        """ Generate Experience Replay Samples """

        warmupSteps = 10000
        state, _ = env.reset()
        for _ in range(warmupSteps):
            action = env.action_space.sample()
            nextState, _, done, _, _ = env.step(action)
            state = nextState
            if done:
                state, _ = env.reset()


        self.experienceReplaySamples = []
        for _ in range(numSamples):
            action = env.action_space.sample()
            nextState, reward, done, _, _ = env.step(action)
            self.experienceReplaySamples.append((state, action, reward, nextState, done))
            state = nextState
            if done:
                state, _ = env.reset()



    def extractExperienceReplayBatch(self, batchSize : int):
        """ Extract Experience Replay Batch """

        if batchSize > len(self.experienceReplaySamples):
            raise ValueError('Not enough samples in the Experience Replay List')

        # replace=True inseamna ca putem alege acelasi sample de mai multe ori
        return np.random.choice(self.experienceReplaySamples, size=batchSize, replace=False)
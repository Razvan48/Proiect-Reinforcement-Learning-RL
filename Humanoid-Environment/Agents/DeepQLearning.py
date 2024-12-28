import gymnasium as gym
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import clone_model




class DeepQLearning:



    def __init__(self):
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.1

        self.experienceReplaySamples = []

        self.MIN_VALUE_ACTION = -0.4
        self.MAX_VALUE_ACTION = 0.4
        self.NUM_BINS_ACTION = 4

        self.ENV_OBSERVATION_SPACE = 348
        self.ENV_ACTION_SPACE = 17

        self.INPUT_DIM = self.ENV_OBSERVATION_SPACE;
        self.OUTPUT_DIM = self.ENV_ACTION_SPACE * self.NUM_BINS_ACTION

        self.QneuralNetwork = Sequential(
            [
                Dense(2 * self.INPUT_DIM, input_dim=self.INPUT_DIM, activation='relu'),
                Dense(2 * self.INPUT_DIM, activation='relu'),
                Dense(self.OUTPUT_DIM, activation='linear')
            ]
        )
        self.QneuralNetwork.compile(optimizer='adam', loss='mse')

        self.targetQneuralNetwork = clone_model(self.QneuralNetwork)
        self.targetQneuralNetwork.set_weights(self.QneuralNetwork.get_weights())
        self.targetQneuralNetwork.compile(optimizer='adam', loss='mse')

        self.EPSILON = 0.1

        self.GAMMA = 0.95



    def simplifyState(self, state):
        return np.round(state, decimals=1)



    def convertFromBinToValue(self, bin):
        binSize = (self.MAX_VALUE_ACTION - self.MIN_VALUE_ACTION) / self.NUM_BINS_ACTION
        value = bin * binSize + self.MIN_VALUE_ACTION + binSize / 2
        return value



    def convertFromValueToBin(self, value):
        DELTA = 0.0001
        binSize = (self.MAX_VALUE_ACTION - self.MIN_VALUE_ACTION) / self.NUM_BINS_ACTION
        bin = (value - self.MIN_VALUE_ACTION) / binSize - DELTA
        return int(bin)



    def convertNetworkOutputToActions(self, output):
        actions = []
        for i in range(len(output)):
            value = self.convertFromBinToValue(i % self.NUM_BINS_ACTION)
            action = np.zeros(self.ENV_ACTION_SPACE)
            action[i // self.NUM_BINS_ACTION] = value
            actions.append((action, output[i]))
        return actions



    def getEpsilonGreedyAction(self, state, env):
        state = self.simplifyState(state)
        output = self.QneuralNetwork.predict(state)[0]
        actions = self.convertNetworkOutputToActions(output)
        if np.random.rand() < self.EPSILON:
            return np.random.choice(actions)[0]
        else:
            return max(actions, key=(lambda x: x[1]))[0]



    def generateExperienceReplaySamples(self, numSamples : int, env : gym.Env):
        """ Generate Experience Replay Samples """

        WARMUP_STEPS = 10000
        state, _ = env.reset()
        for _ in range(WARMUP_STEPS):
            action = env.action_space.sample()
            nextState, _, done, _, _ = env.step(action)
            state = nextState
            if done:
                state, _ = env.reset()


        self.experienceReplaySamples = []
        for _ in range(numSamples):
            action = self.getEpsilonGreedyAction(state, env)
            nextState, reward, done, _, _ = env.step(action)
            self.experienceReplaySamples.append((self.simplifyState(state), action, reward, self.simplifyState(nextState), done))
            state = nextState
            if done:
                state, _ = env.reset()



    def extractExperienceReplayBatch(self, batchSize : int):
        """ Extract Experience Replay Batch """

        if batchSize > len(self.experienceReplaySamples):
            raise ValueError('Not enough samples in the Experience Replay List')

        # replace=True inseamna ca putem alege acelasi sample de mai multe ori
        return np.random.choice(self.experienceReplaySamples, size=batchSize, replace=False)



    def getActionIndexInOutput(self, action): # se presupune ca acest action e unul simplificat (are un singur element nenul)
        for i in range(len(action)):
            if action[i] != 0.0:
                return i * self.NUM_BINS_ACTION + self.convertFromValueToBin(action[i])



    def train(self, env):
        NUM_ITERATIONS = 1000
        NUM_SUBITERATIONS = 50

        NUM_SAMPLES = 10000
        BATCH_SIZE = 256

        NUM_ITERATIONS_UNTIL_TARGET_UPDATE = 100

        for numIteration in range(NUM_ITERATIONS):
            print('Iteration:', numIteration)

            self.generateExperienceReplaySamples(NUM_SAMPLES, env)

            for numSubiteration in range(NUM_SUBITERATIONS):
                print('Subiteration:', numSubiteration)

                batch = self.extractExperienceReplayBatch(BATCH_SIZE)

                for state, action, reward, nextState, done in batch:
                    qValues = self.QneuralNetwork.predict(state)[0]
                    qValue = qValues[self.getActionIndexInOutput(action)]
                    targetValues = self.targetQneuralNetwork.predict(nextState)[0]
                    targetValue = np.max(targetValues)

                    targetQValue = reward + self.GAMMA * targetValue

                    qValues[self.getActionIndexInOutput(action)] = targetQValue
                    self.QneuralNetwork.fit(state, qValues, verbose=0)

            if numIteration % NUM_ITERATIONS_UNTIL_TARGET_UPDATE == 0:
                print('Updating target network...')

                self.targetQneuralNetwork.set_weights(self.QneuralNetwork.get_weights())
                self.targetQneuralNetwork.compile(optimizer='adam', loss='mse')



    def choose_action(self, state):
        state = self.simplifyState(state)
        output = self.targetQneuralNetwork.predict(state)[0]
        actions = self.convertNetworkOutputToActions(output)
        return max(actions, key=(lambda x: x[1]))[0]

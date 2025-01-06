import gymnasium as gym
import numpy as np
import os

import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.models import clone_model

import Helper



class DeepQLearning:



    def __init__(self):

        self.experienceReplaySamples = []

        self.MIN_VALUE_ACTION = -0.4
        self.MAX_VALUE_ACTION = 0.4
        self.NUM_BINS_ACTION = 4

        self.ENV_OBSERVATION_SPACE = 348
        self.ENV_ACTION_SPACE = 17

        self.INPUT_DIM = self.ENV_OBSERVATION_SPACE
        self.OUTPUT_DIM = self.ENV_ACTION_SPACE * self.NUM_BINS_ACTION

        self.qNeuralNetwork = tf.keras.models.Sequential([
                tf.keras.layers.Dense(2 * self.INPUT_DIM, input_dim=self.INPUT_DIM, activation='relu'),
                tf.keras.layers.Dense(2 * self.INPUT_DIM, activation='relu'),
                tf.keras.layers.Dense(self.OUTPUT_DIM, activation='linear')
            ])
        self.qNeuralNetwork.compile(optimizer='adam', loss='mse')

        self.targetQNeuralNetwork = tf.keras.models.clone_model(self.qNeuralNetwork)
        self.targetQNeuralNetwork.set_weights(self.qNeuralNetwork.get_weights())
        self.targetQNeuralNetwork.compile(optimizer='adam', loss='mse')

        self.START_EPSILON = 1.0
        self.EPSILON_DECAY_RATE = 0.999
        self.MIN_EPSILON = 0.1
        self.EPSILON = self.START_EPSILON

        self.START_GAMMA = 0.95
        self.GAMMA_DECAY_RATE = 0.999
        self.MIN_GAMMA = 0.1
        self.GAMMA = self.START_GAMMA



    def simplifyState(self, state : np.ndarray) -> np.ndarray:
        return np.round(state, decimals=1)



    def convertFromBinToValue(self, bin : int) -> float:
        binSize = (self.MAX_VALUE_ACTION - self.MIN_VALUE_ACTION) / self.NUM_BINS_ACTION
        value = bin * binSize + self.MIN_VALUE_ACTION + binSize / 2
        return value



    def convertFromValueToBin(self, value : float) -> int:
        DELTA = 0.0001
        binSize = (self.MAX_VALUE_ACTION - self.MIN_VALUE_ACTION) / self.NUM_BINS_ACTION
        bin = (value - self.MIN_VALUE_ACTION) / binSize - DELTA
        return int(bin)



    def convertNetworkOutputToActions(self, output : np.ndarray):
        actions = []
        scores = []
        for i in range(len(output)):
            value = self.convertFromBinToValue(i % self.NUM_BINS_ACTION)
            action = [0.0 for _ in range(self.ENV_ACTION_SPACE)]
            action[i // self.NUM_BINS_ACTION] = value

            actions.append(action)
            scores.append(output[i])
        return (actions, scores)



    def getEpsilonGreedyAction(self, state):
        state = self.simplifyState(state)
        output = self.qNeuralNetwork.predict(state.reshape(1, self.INPUT_DIM))[0]
        actions, scores = self.convertNetworkOutputToActions(output)
        if np.random.rand() < self.EPSILON:
            return actions[np.random.randint(len(actions))]
        else:
            return actions[np.argmax(scores)]



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
            action = self.getEpsilonGreedyAction(state)
            nextState, reward, done, _, _ = env.step(action)
            self.experienceReplaySamples.append((self.simplifyState(state), action, reward, self.simplifyState(nextState), done)) # state, action sunt simplificate
            state = nextState
            if done:
                state, _ = env.reset()



    def extractExperienceReplayBatch(self, batchSize : int):
        """ Extract Experience Replay Batch """

        if batchSize > len(self.experienceReplaySamples):
            raise ValueError('Not enough samples in the Experience Replay List')

        # replace=True inseamna ca putem alege acelasi sample de mai multe ori
        return np.random.choice(self.experienceReplaySamples, size=batchSize, replace=False)



    def getActionIndexInOutput(self, action): # Se presupune ca acest action e unul simplificat (are un singur element nenul)
        for i in range(len(action)):
            if action[i] != 0.0:
                return i * self.NUM_BINS_ACTION + self.convertFromValueToBin(action[i])



    def updateEpsilon(self, numIteration : int) -> float:
        return max(self.MIN_EPSILON, self.START_EPSILON * (self.EPSILON_DECAY_RATE ** (numIteration // 10)))



    def updateGamma(self, numIteration : int) -> float:
        return max(self.MIN_GAMMA, self.START_GAMMA * (self.GAMMA_DECAY_RATE ** (numIteration // 10)))



    def train(self, env):
        NUM_ITERATIONS = 1000
        NUM_SUBITERATIONS = 10

        NUM_SAMPLES = 20000
        BATCH_SIZE = 1024

        NUM_ITERATIONS_UNTIL_TARGET_UPDATE = 100

        self.EPSILON = self.START_EPSILON
        self.GAMMA = self.START_GAMMA

        rewardsHistory = []
        RECENT_REWARDS_HISTORY_SIZE = 100
        NUM_ITERATIONS_UNTIL_REWARDS_PRINT = 100

        for numIteration in range(NUM_ITERATIONS):
            print('Iteration:', numIteration)

            self.generateExperienceReplaySamples(NUM_SAMPLES, env)

            for numSubiteration in range(NUM_SUBITERATIONS):
                print('Subiteration:', numSubiteration)

                batch = self.extractExperienceReplayBatch(BATCH_SIZE) # state, action sunt simplificate

                ''' Codul acesta face fit pas cu pas
                for state, action, reward, nextState, done in batch:
                    qValues = self.qNeuralNetwork.predict(state.reshape(1, self.INPUT_DIM))[0]
                    qValue = qValues[self.getActionIndexInOutput(action)]
                    targetValues = self.targetQNeuralNetwork.predict(nextState.reshape(1, self.INPUT_DIM))[0]
                    targetValue = np.max(targetValues)

                    targetQValue = reward + self.GAMMA * targetValue

                    qValues[self.getActionIndexInOutput(action)] = targetQValue
                    self.qNeuralNetwork.fit(state, qValues, verbose=0)
                    
                    rewardsHistory.append(reward)
                '''

                # Cod optimizat
                states = np.array([state for state, _, _, _, _ in batch])
                nextStates = np.array([nextState for _, _, _, nextState, _ in batch])
                qValues = self.qNeuralNetwork.predict(states)
                targetValues = self.targetQNeuralNetwork.predict(nextStates)

                for i, (state, action, reward, nextState, done) in enumerate(batch):
                    qValue = qValues[i][self.getActionIndexInOutput(action)]
                    targetValue = np.max(targetValues[i])

                    targetQValue = reward + self.GAMMA * targetValue
                    qValues[i][self.getActionIndexInOutput(action)] = targetQValue

                    rewardsHistory.append(reward)

                self.qNeuralNetwork.fit(states, qValues, verbose=0)


            if numIteration % NUM_ITERATIONS_UNTIL_TARGET_UPDATE == 0:
                print('Updating target network...')

                self.targetQNeuralNetwork.set_weights(self.qNeuralNetwork.get_weights())
                self.targetQNeuralNetwork.compile(optimizer='adam', loss='mse')

            self.EPSILON = self.updateEpsilon(numIteration)
            self.GAMMA = self.updateGamma(numIteration)

            if numIteration % NUM_ITERATIONS_UNTIL_REWARDS_PRINT and len(rewardsHistory) >= RECENT_REWARDS_HISTORY_SIZE:
                averageReward = np.mean(rewardsHistory[-RECENT_REWARDS_HISTORY_SIZE:])
                maximumReward = np.max(rewardsHistory[-RECENT_REWARDS_HISTORY_SIZE:])
                print(f'Average Reward: {averageReward:.3f} - Maximum Reward: {maximumReward:.3f}')



    def choose_action(self, state):
        state = self.simplifyState(state)
        output = self.targetQNeuralNetwork.predict(state.reshape(1, self.INPUT_DIM))[0]
        actions, scores = self.convertNetworkOutputToActions(output)
        return actions[np.argmax(scores)]



    def save_model(self, file_name):
        DIRECTORY_PATH = 'Data/DeepNetworks/'
        os.makedirs(DIRECTORY_PATH, exist_ok=True)
        self.targetQNeuralNetwork.save(DIRECTORY_PATH + file_name)
        # self.targetQNeuralNetwork.compile(optimizer='adam', loss='mse')

        self.qNeuralNetwork = tf.keras.models.clone_model(self.targetQNeuralNetwork)
        self.qNeuralNetwork.set_weights(self.targetQNeuralNetwork.get_weights())
        self.qNeuralNetwork.compile(optimizer='adam', loss='mse')



    def load_model(self, file_name, env):
        DIRECTORY_PATH = 'Data/DeepNetworks/'
        self.targetQNeuralNetwork = tf.keras.models.load_model(DIRECTORY_PATH + file_name)
        self.targetQNeuralNetwork.compile(optimizer='adam', loss='mse')

        self.qNeuralNetwork = tf.keras.models.clone_model(self.targetQNeuralNetwork)
        self.qNeuralNetwork.set_weights(self.targetQNeuralNetwork.get_weights())
        self.qNeuralNetwork.compile(optimizer='adam', loss='mse')



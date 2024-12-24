import numpy as np
import gymnasium as gym
from collections import defaultdict
from Helper import Helper as hp
from Configuration import Configuration as Conf

class MonteCarlo:
    def __init__(self, gamma=0.95, epsilon=1.0, num_episodes=5000, epsilon_decay=0.999, min_epsilon=0.1):
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.policy = {} 
        self.Q = defaultdict()
        self.Q_n = defaultdict()
        self.Q_actions = defaultdict()
        self.helper = hp.Helper()
        self.conf = Conf.Configuration()
        self.action_low = None  
        self.action_high = None  
        self.state_space = None 
        self.action_space = None
        self.UPDATE_POLICY_EVERY = self.conf.UPDATE_POLICY_EVERY

    def initialize_policy(self, env):
        """Initialize policy for continuous action space."""
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high
        self.action_space = env.action_space.shape[0]   
        self.state_space = env.observation_space.shape[0]
        

    def choose_action(self, state):
        """Choose an action based on the epsilon-greedy policy."""
        state_key = self.helper.discretize_state(state)
        
        if state_key not in self.policy:
            self.policy[state_key] = self.helper.discretize_action(np.random.uniform(self.action_low, self.action_high))

        if np.random.rand() > self.epsilon:
            action = self.policy[state_key] 
        else:
            action = self.helper.discretize_action(np.random.uniform(self.action_low, self.action_high))

        return self.helper.reverse_discretize_action(action)

    def monte_carlo_policy(self, episode, firstVisit = True):
        """Update policy using Monte Carlo returns."""

        first_visit_state_actions = {}

        if firstVisit == True:
            for t in range(len(episode)):   
                state, action, reward = episode[t]
                state_key = self.helper.discretize_state(state)
                action_key = self.helper.discretize_action(action)

                if (state_key, action_key) not in first_visit_state_actions:
                    first_visit_state_actions[(state_key, action_key)] = t

        G = 0  
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            state_key = self.helper.discretize_state(state)

            action = self.helper.discretize_action(action)

            if firstVisit == True:
                if first_visit_state_actions[(state_key, action)] < t:
                    continue

            G = reward + self.gamma * G 

            if state_key not in self.Q_actions:
                self.Q_actions[state_key] = set()
            self.Q_actions[state_key].add(action)

            if state_key not in self.Q_n:
                self.Q_n[state_key] = {}

            if action not in self.Q_n[state_key]:
                self.Q_n[state_key][action] = 0

            if state_key not in self.Q:
                self.Q[state_key] = {}

            if action not in self.Q[state_key]:
                self.Q[state_key][action] = 0


            self.Q_n[state_key][action] += 1
            alpha = min(1, 1.0 / self.Q_n[state_key][action] * (self.Q_n[state_key][action] / 3))
            self.Q[state_key][action] = self.Q[state_key][action] + alpha * (G - self.Q[state_key][action])

    def update_policy(self):
        """Update policy based on the Q values."""
        for state_key in self.Q_actions:
            Max = -1e9
            best_action = None
            for action_in_Q in self.Q_actions[state_key]:
                if self.Q[state_key][action_in_Q] > Max:
                    Max = self.Q[state_key][action_in_Q]
                    best_action = action_in_Q
            
            self.policy[state_key] = best_action

    def generate_episode(self, env):
        """Generate an episode using the current policy."""
        episode = []
        state = env.reset()[0]
        done = False

        while not done:
            action = self.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        return episode

    def train(self, env):
        """Train the agent using the Monte Carlo method."""
        self.initialize_policy(env)
        rewards_history = [] 

        for episode in range(1, self.num_episodes + 1):
            episode_data = self.generate_episode(env)
            total_reward = sum([reward for _, _, reward in episode_data])
            rewards_history.append(total_reward)
            
            self.monte_carlo_policy(episode_data, False)
            
            if episode % self.UPDATE_POLICY_EVERY == 0:
                self.update_policy()

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            if episode % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                max_reward = np.max(rewards_history[-100:])
                print(f"Episode {episode} complete - Average Reward: {avg_reward:.2f} - Max Reward: {max_reward:.2f}")
        
        self.update_policy()
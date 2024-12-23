import numpy as np
import gymnasium as gym
from Helper import Helper as hp

class MonteCarlo:
    def __init__(self, gamma=0.99, epsilon=0.1, num_episodes=5000):
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.policy = {} 
        self.returns = {} 
        self.helper = hp.Helper()
        self.action_low = None  
        self.action_high = None  
        self.state_space = None 
        self.action_space = None

    def initialize_policy(self, env):
        """Initialize policy for continuous action space."""
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high
        self.action_space = env.action_space.shape[0]   
        self.state_space = env.observation_space.shape[0]
        
        self.policy = {
            state: (np.zeros(env.action_space.shape), np.ones(env.action_space.shape) * 0.4)
            for state in range(self.state_space)
        }

    def choose_action(self, state):
        """Choose an action based on the epsilon-greedy policy."""
        state_key = self.helper.discretize_state(state)
        
        if state_key not in self.policy:
            self.policy[state_key] = (np.zeros(self.action_low.shape), np.ones(self.action_low.shape) * 0.4)
        
        mean, std_dev = self.policy[state_key]
        
        if np.random.rand() > self.epsilon:
            action = np.random.normal(mean, std_dev)
        else:
            action = np.random.uniform(self.action_low, self.action_high)

        return np.clip(action, self.action_low, self.action_high)

    def monte_carlo_policy_update(self, episode):
        """Update policy using Monte Carlo returns."""
        G = 0  # Initialize return
        visited_state_actions = set()

        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            state_key = self.helper.discretize_state(state)
            G = reward + self.gamma * G
            
            action = tuple(action)

            if (state_key, action) not in visited_state_actions:
                visited_state_actions.add((state_key, action))
                if (state_key, action) not in self.returns:
                    self.returns[(state_key, action)] = []
                self.returns[(state_key, action)].append(G)

                # Update policy for the state
                action_values = [
                    np.mean(self.returns[(state_key, a)]) if (state_key, a) in self.returns else 0
                    for a in range(self.action_space)
                ]
                self.policy[state_key] = np.argmax(action_values)

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
        self.initialize_policy(env)
        rewards_history = [] 

        for episode in range(1, self.num_episodes + 1):
            episode_data = self.generate_episode(env)
            total_reward = sum([reward for _, _, reward in episode_data])
            rewards_history.append(total_reward)
            
            self.monte_carlo_policy_update(episode_data)

            if episode % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                print(f"Episode {episode} complete - Average Reward: {avg_reward:.2f}")
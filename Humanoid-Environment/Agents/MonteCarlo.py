import gymnasium as gym

class MonteCarlo:
    def __init__(self):
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.memory = []
        self.model = self.create_model()
        
    def create_model(self):
        pass
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def choose_action(self, obs):
        pass
        
    def replay(self):
        pass
        
    def save(self):
        pass
        
    def load(self):
        pass
    
    def train(self, env : gym.Env):
        pass

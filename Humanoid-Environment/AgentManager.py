from Agents import DeepQLearning as DQL
from Agents import MonteCarlo as MC
from Agents import ProximalPolicyOptimization as PPO
import gymnasium as gym

class AgentManager:
    def __init__(self, agent_name : str):
        self.agent_name = agent_name
        self.agent = self.create_agent()
        
    def create_agent(self):
        if self.agent_name == 'DeepQLearningAgent':
            return DQL.DeepQLearning()
        elif self.agent_name == 'MonteCarloAgent':
            return MC.MonteCarlo()
        elif self.agent_name == 'ProximalPolicyOptimizationAgent':
            return PPO.ProximalPolicyOptimization()
        else:
            raise ValueError('Agent not found')
            
    def train_agent(self, env : gym.Env):
        self.agent.train(env)
        
    def choose_action(self, obs):
        return self.agent.choose_action(obs)
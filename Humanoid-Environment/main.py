import gymnasium as gym
import sys
import AgentManager as AM
from Configuration import Configuration as Conf
from Helper import Helper as hp

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <AgentName> where AgentName is one of the following: DeepQLearningAgent, MonteCarloAgent")
        sys.exit(1)
    
    # create environment for Humanoid-v5
    env = gym.make("Humanoid-v5", render_mode="human")

    # define the agent to be used and train it based on the environment
    agent = AM.AgentManager(sys.argv[1])
    agent.train_agent(env)
    
    # declare general configuration
    conf = Conf.Configuration()
    helper = hp.Helper()
    
    # test the agent
    state, info = env.reset(seed=0)
    for _ in range(conf.NUM_ITERATIONS):
        # TODO - update choose_action method in MonteCarlo.py and DeepQLearning.py
        state = helper.discretize_state(state)
        action = agent.choose_action(state)
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        env.render()
    
    env.close()
    
if __name__ == "__main__":
    main()
    
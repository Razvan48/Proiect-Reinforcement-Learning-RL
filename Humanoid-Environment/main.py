import gymnasium as gym
import sys
import AgentManager as AM
from Configuration import Configuration as Conf

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
    
    # test the agent on the environment
    state, info = env.reset(seed=0)
    for _ in range(conf.NUM_ITERATIONS):
        # TODO - update choose_action method in MonteCarlo.py and DeepQLearning.py
        action = agent.choose_action(state)
        state, reward, done, truncated, info = env.step(action)

        env.render()
    
    env.close()
    
if __name__ == "__main__":
    main()
    
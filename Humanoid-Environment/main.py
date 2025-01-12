import gymnasium as gym
import sys
import AgentManager as AM
from Configuration import Configuration as Conf
from matplotlib import pyplot as plt

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <AgentName> where AgentName is one of the following: DeepQLearningAgent, MonteCarloAgent, ProximalPolicyOptimizationAgent")
        sys.exit(1)

    # create configuration object
    conf = Conf.Configuration()

    # create environment for Humanoid-v5
    env = gym.make("Humanoid-v5", render_mode="human")

    # define the agent to be used and train it based on the environment
    agent = AM.AgentManager(sys.argv[1])

    # load model if it exists
    file = input("Enter the file name to load the model: (Press Enter to skip) ")
    if file:
        agent.load_model(file, env)

    file = input("Enter the number of episodes for training: (Type 0 to skip) ")
    if file:
        conf.N_EPISODES = int(file)

    agent.train_agent(env)

    # Test the agent on the environment
    # Store performance metrics
    total_reward = [0] * conf.N_EPISODES_TEST  # Initialize the list with zeros

    for episode in range(conf.N_EPISODES_TEST):
        done = False
        state, info = env.reset()  # Ensure the environment reset returns 'state'
        while not done:
            # TODO - update choose_action method in MonteCarlo.py and DeepQLearning.py
            action = agent.choose_action(state)
            state, reward, done, truncated, info = env.step(action)

            total_reward[episode] += reward  # Add the reward for the current episode

            env.render()  # Render the environment

        print(f"Total Reward for Episode {episode + 1}: {total_reward[episode]}")

    # Calculate and print the average reward
    average_reward = sum(total_reward) / len(total_reward) if total_reward else 0
    print("Average Reward: ", average_reward)

    env.close()

    plt.plot(total_reward)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode" + " - Average Reward: " + str(average_reward))
    plt.show()

    # save the model
    print("Saving model...")
    file = input("Enter the file name to save the model: (Press Enter to skip) ")
    if file:
        agent.save_model(file)

if __name__ == "__main__":
    main()
    
import gymnasium as gym

env = gym.make("Humanoid-v5", render_mode="human")

obs, info = env.reset(seed=0)

print('Initial Observation Shape:', obs.shape)
print('Initial Observation:', obs)

print('Action Space Shape:', env.action_space.shape)
print('Action Space:', env.action_space)

print('Observation Space Shape:', env.observation_space.shape)
print('Observation Space:', env.observation_space)


NUM_ITERATIONS = 1000
for _ in range(NUM_ITERATIONS):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

    env.render()




env.close()

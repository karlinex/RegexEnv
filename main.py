from RegexEnv import RegexEnv

text_length = 30
env = RegexEnv(text_length)
state_shape = env.observation_space.shape
for t in range(text_length):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
env.close()
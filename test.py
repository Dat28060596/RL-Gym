import gym
import keras
import numpy as np
from keras.losses import mean_squared_error

# Ensure custom objects are registered
keras.utils.register_keras_serializable('Custom', 'mse')(mean_squared_error)

# Create environment
env = gym.make("CartPole-v1", render_mode="human")
state, _ = env.reset()
state_size = env.observation_space.shape[0]

# Load trained agent with custom objects
custom_objects = {'mse': mean_squared_error}
my_agent = keras.models.load_model(r"C:\Users\Admin\PycharmProjects\RL_LEARNING\train_agent.h5", custom_objects=custom_objects)

n_timesteps = 500
total_reward = 0

for t in range(n_timesteps):
    env.render()
    # Reshape state and predict action
    state = state.reshape((1, state_size))
    q_values = my_agent.predict(state, verbose=0)
    max_q_values = np.argmax(q_values)

    # Step through environment
    next_state, reward, terminal, _, _ = env.step(action=max_q_values)
    total_reward += reward
    state = next_state
    print(t)

env.close()

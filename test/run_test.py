import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the actor and critic models
actor = keras.models.load_model('models/actor.h5')
critic = keras.models.load_model('models/critic.h5')

# Initialize the environment
env = gym.make('CartPole-v1')

# Run the environment
observation = env.reset()
done = False

while not done:
    # Reshape the observation and pass it through the actor model to get the action logits
    observation = np.reshape(observation, (1, -1))
    action_logits = actor.predict(observation)
    
    # Sample an action from the action logits
    action = np.argmax(action_logits)

    # Take a step in the environment
    observation, reward, done, _ = env.step(action)

    # Render the environment (optional)
    env.render()

# Close the environment
env.close()


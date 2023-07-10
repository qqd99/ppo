import tensorflow as tf
from tensorflow.keras import layers, Input, Model
import keras
import pickle
import numpy as np
import cv2
from sklearn.model_selection import train_test_split


def load_pickle_file(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

# Example usage:
file_path = "/content/drive/MyDrive/!/traj_0.pkl"
loaded_data = load_pickle_file(file_path)

observations = loaded_data['observations']
observations = tf.cast(observations, dtype=tf.float32) / 255.0
observations = tf.reshape(observations, (-1, 256, 256, 3))
observations = tf.image.resize(observations, size=(84, 84))

actions = loaded_data['actions']
num_of_point = tf.minimum(len(observations), len(actions))

unique_actions, encoded_actions = np.unique(actions, return_inverse=True)
one_hot_encoded_actions =  tf.one_hot(encoded_actions, depth=len(unique_actions))

k = 10  # Length of each segment

num_splits = num_of_point// k
num_splits = num_splits.numpy()

obs_segments = tf.split(observations[:num_splits * k], num_splits)
act_segments = tf.split(one_hot_encoded_actions[:num_splits * k], num_splits)


segments = [tf.concat((tf.reshape(obs, [-1]), tf.reshape(act, [-1])), axis=0) for obs, act in zip(obs_segments, act_segments)]

left_seg = []
right_seg = []
labels = []

selected_combinations = set()

while len(selected_combinations) <2000:
    # Randomly select two unique segments
    indices = np.random.choice(len(segments), size=2, replace=False)
    combination = tuple(sorted(indices))

    if combination in selected_combinations:
        continue

    # Add the combination to the selected combinations
    selected_combinations.add(combination)

    # Retrieve the selected segments based on the indices

    left_seg.append(segments[indices[0]])
    right_seg.append(segments[indices[1]])
    labels.append(np.random.choice([1, 0]))
                
left_seg = tf.convert_to_tensor(left_seg)
right_seg = tf.convert_to_tensor(right_seg)
labels = tf.convert_to_tensor(labels)

  
# Input is obs-act pair
class FullyConnectedMLP(layers.Layer):
    """Vanilla two hidden layer multi-layer perceptron"""

    def __init__(self, h_size=64):
        super(FullyConnectedMLP, self).__init__()

        self.model = tf.keras.Sequential([
            layers.Dense(h_size),
            layers.LeakyReLU(),
            layers.Dropout(0.5),
            layers.Dense(h_size),
            layers.LeakyReLU(),
            layers.Dropout(0.5),
            layers.Dense(1)
        ])

    def call(self, segments):
        return self.model(segments)

class RewardModel(keras.Model):
    def __init__(self,obs_shape, act_shape):
        super().__init__()

        self.reward = FullyConnectedMLP()

    def call(self, segments, training=False):

        reward_1 = self.reward(segments[0] , training=training)
        reward_2 = self.reward(segments[1], training=training)
        return tf.concat([reward_1, reward_2], axis=-1)

model =  RewardModel(observations.shape[-3:], unique_actions.shape)

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
print("Start training")
# Train the model
model.fit([left_seg, right_seg], labels, epochs=20, batch_size=64)
print("End training")

import tensorflow as tf
from tensorflow.keras import layers, Input, Model

# Input is obs-act pair
class FullyConnectedMLP(layers.Layer):
    """Vanilla two hidden layer multi-layer perceptron"""

    def __init__(self, obs_shape, act_shape, h_size=64):
        super(FullyConnectedMLP, self).__init__()
        input_dim = np.prod(obs_shape) + np.prod(act_shape)

        self.model = tf.keras.Sequential([
            keras.Input(shape = input_dim),
            layers.Dense(h_size),
            layers.LeakyReLU(),
            layers.Dropout(0.5),
            layers.Dense(h_size),
            layers.LeakyReLU(),
            layers.Dropout(0.5),
            layers.Dense(1)
        ])

    def run(self, obs, act):
        flat_obs = tf.keras.layers.Flatten()(obs)
        x = tf.concat([flat_obs, act], axis=1)
        return self.model(x)

class reward_model(keras.Model):
    def __init__(self):
        super(ResNet_Like, self).__init__()

        self.reward = FullyConnectedMLP(obs_shape, act_shape)

    def call(self, segments, training=False):
        segment_1 = segments[0]
        segment_2 = segments[1]
        
        self.obs_1 = tf.reshape(segment_1["obs"], (-1,) + self.obs_shape)
        self.acts_1 = tf.reshape(segment_1["acts"], (-1,) + self.act_shape)
        self.obs_2 = tf.reshape(segment_1["obs"], (-1,) + self.obs_shape)
        self.acts_2 = tf.reshape(segment_1["acts"], (-1,) + self.act_shape)
        
        reward_1 = self.reward(self.obs_1,self.acts_1 , training=training)
        reward_2 = self.reward(self.obs_2,self.acts_2 , training=training)

        return tf.concat([output_1, output_2], axis=-1)


# Compile the model
model.compile(optimizer=keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit([segment_1, segment_2], labels, epochs=20, batch_size=64)

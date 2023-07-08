import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import numpy as np
import time
import gym
import scipy.signal
import time
from collections import deque
import pyautogui
import utils


#-----------------------Actor------------------------

class Actor(tf.keras.Model):
    """MLP actor network."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters
    ):
        super(Actor, self).__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform(), input_shape=(encoder_feature_dim,)),
            tf.keras.layers.Dense(hidden_dim, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform()),
            tf.keras.layers.Dense(2 * action_shape, kernel_initializer=tf.keras.initializers.GlorotUniform())
        ])

        self.outputs = {}


    def call(
        self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False
    ):
        obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = tf.split(self.trunk(obs), num_or_size_splits=2, axis=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = tf.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = tf.exp(log_std)

        if compute_pi:
            std = tf.exp(log_std)
            noise = tf.random.normal(shape=tf.shape(mu))
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = utils.gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = utils.squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std, entropy



#-----------------------Critic------------------------

class Critic(tf.keras.Model):
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters
    ):
        super(Critic, self).__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )

        self.Q1 = QFunction(
            self.encoder.feature_dim, action_shape, hidden_dim
        )
        self.Q2 = QFunction(
            self.encoder.feature_dim, action_shape, hidden_dim
        )

        self.outputs = dict()

    def call(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propagation to encoder
        obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2


#-----------------------QFunction------------------------
class QFunction(tf.keras.Model):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super(QFunction, self).__init__()

        self.trunk = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation=tf.nn.relu,kernel_initializer=tf.keras.initializers.GlorotUniform(), input_dim=obs_dim+action_dim),
            tf.keras.layers.Dense(hidden_dim, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform()),
            tf.keras.layers.Dense(1)
        ])

    def call(self, obs, action):
        assert obs.shape[0] == action.shape[0]

        obs_action = tf.concat([obs, action], axis=1)
        return self.trunk(obs_action)


#-----------------------CURL------------------------
    
class CURL(tf.keras.Model):
    def __init__(self, obs_shape, z_dim, batch_size, critic, critic_target, output_type="continuous"):
        super(CURL, self).__init__()
        self.batch_size = batch_size

        self.encoder = critic.encoder
        self.encoder_target = critic_target.encoder

        self.W = tf.Variable(tf.random.uniform(shape=(z_dim, z_dim)))
        self.output_type = output_type

    def encode(self, x, detach=False, ema=False):
        if ema:
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)

        if detach:
            z_out = tf.stop_gradient(z_out)
        return z_out

    def compute_logits(self, z_a, z_pos):
        Wz = tf.matmul(self.W, z_pos, transpose_b=True)
        logits = tf.matmul(z_a, Wz)
        logits = logits - tf.reduce_max(logits, axis=1, keepdims=True)
        return logits


#-----------------------CurlPPOAgent------------------------

class CurlPPOAgent(object):
    """CURL representation learning with SAC."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        num_layers=4,
        num_filters=32,
        cpc_update_freq=1,
        detach_encoder=False,
        curl_latent_dim=128
    ):

        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.cpc_update_freq = cpc_update_freq
        self.image_size = obs_shape[-1]
        self.curl_latent_dim = curl_latent_dim
        self.detach_encoder = detach_encoder
        self.encoder_type = encoder_type

        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters
        )

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        )

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        )

        self.critic_target.set_weights(self.critic.trainable_variables)

        # Tie encoders between actor and critic, and CURL and critic
        self.actor.encoder.set_weights(self.critic.encoder.trainable_variables)

        self.log_alpha = tf.Variable(np.log(init_temperature), trainable=True, dtype=tf.float32)
        self.target_entropy = -tf.reduce_prod(action_shape)

        # Initialize optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr, beta_1=actor_beta)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr, beta_1=critic_beta)
        self.log_alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha_lr, beta_1=alpha_beta)

        if encoder_type == 'pixel':
            # Create CURL encoder
            self.CURL = CURL(obs_shape, encoder_feature_dim, self.curl_latent_dim,
                            self.critic, self.critic_target, output_type='continuous')

            # Optimizers for critic encoder and CURL
            self.encoder_optimizer = tf.keras.optimizers.Adam(learning_rate=encoder_lr)
            self.cpc_optimizer = tf.keras.optimizers.Adam(learning_rate=encoder_lr)

        self.cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy()

    def train(self, training=True):
        self.training = training
        self.actor.trainable = training
        self.critic.trainable = training
        if self.encoder_type == 'pixel':
            self.CURL.trainable = training

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)
    
    def select_action(self, obs):
        obs = tf.expand_dims(tf.convert_to_tensor(obs, dtype=tf.float32), axis=0)
        mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False, training=False)
        return mu.numpy().flatten()

    def sample_action(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = utils.center_crop_image(obs, self.image_size)

        obs = tf.expand_dims(tf.convert_to_tensor(obs, dtype=tf.float32), axis=0)
        mu, pi, _, _ = self.actor(obs, compute_log_pi=False, training=False)
        return pi.numpy().flatten()

    @tf.function
    def update_critic(self, obs, action, reward, next_obs, not_done):
        with tf.GradientTape() as tape:
            _, policy_action, log_pi, _ = self.actor(next_obs, training=False)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action, training=False)
            target_V = tf.reduce_min([target_Q1, target_Q2], axis=0) - self.alpha * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(obs, action, detach_encoder=self.detach_encoder, training=True)
            critic_loss = tf.keras.losses.MeanSquaredError()(current_Q1, target_Q) + tf.keras.losses.MeanSquaredError()(current_Q2, target_Q)

        # Optimize the critic
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))


    @tf.function
    def update_actor(self, obs):
        with tf.GradientTape(persistent=True) as tape:
            # Compute log-probabilities of the current policy 
            _, _, logprobs, _ = self.actor(obs, detach_encoder=True)

            # Compute ratios and surrogate objectives
            ratios = tf.exp(logprobs - old_logprobs)
            surrogates = ratios * advantages

            # Compute clipped surrogate objective
            clipped_surrogates = tf.clip_by_value(ratios, 1 - clip_ratio, 1 + clip_ratio) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surrogates, clipped_surrogates))

  
        # Optimize the actor
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

    @tf.function
    def update_cpc(self, obs_anchor, obs_pos, cpc_kwargs):
        z_a = self.CURL.encode(obs_anchor)
        z_pos = self.CURL.encode(obs_pos, ema=True)

        logits = self.CURL.compute_logits(z_a, z_pos)
        labels = tf.range(tf.shape(logits)[0], dtype=tf.int64)
        loss = self.cross_entropy_loss(labels, logits)

        encoder_gradients = tape.gradient(loss, self.critic.encoder.trainable_variables)
        cpc_gradients = tape.gradient(loss, self.CURL.trainable_variables)

        self.encoder_optimizer.apply_gradients(zip(encoder_gradients, self.critic.encoder.trainable_variables))
        self.cpc_optimizer.apply_gradients(zip(cpc_gradients, self.CURL.trainable_variables))


    @tf.function
    def update(self, replay_buffer, step):
        if self.encoder_type == 'pixel':
            obs, action, reward, next_obs, not_done, cpc_kwargs = replay_buffer.sample_cpc()
        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample_proprio()

        self.update_critic(obs, action, reward, next_obs, not_done)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1.trainable_variables, self.critic_target.Q1.trainable_variables, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2.trainable_variables, self.critic_target.Q2.trainable_variables, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder.trainable_variables, self.critic_target.encoder.trainable_variables,
                self.encoder_tau
            )

        if step % self.cpc_update_freq == 0 and self.encoder_type == 'pixel':
            obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
            self.update_cpc(obs_anchor, obs_pos, cpc_kwargs)

    def save(self, model_dir, step):
        self.actor.save_weights(f'{model_dir}/actor_{step}.h5')
        self.critic.save_weights(f'{model_dir}/critic_{step}.h5')

    def save_curl(self, model_dir, step):
        self.CURL.save_weights(f'{model_dir}/curl_{step}.h5')

    def load(self, model_dir, step):
        self.actor.load_weights(f'{model_dir}/actor_{step}.h5')
        self.critic.load_weights(f'{model_dir}/critic_{step}.h5')



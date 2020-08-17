import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers

from tf2rl.algos.policy_base import OffPolicyAgent
from tf2rl.misc.target_update_ops import update_target_variables
from tf2rl.misc.huber_loss import huber_loss


class Actor(tf.keras.Model):
    def __init__(self, state_shape, action_space, units=[400, 300], name="Actor"):
        super().__init__(name=name)

        self._action_space = action_space

        self.l1 = Dense(units[0], name="L1", kernel_regularizer=regularizers.L1L2(l1=0, l2=1e-2))
        self.l2 = Dense(units[1], name="L2", kernel_regularizer=regularizers.L1L2(l1=0, l2=1e-2))
        self.l3 = Dense(action_space.high.size, name="L3", kernel_regularizer=regularizers.L1L2(l1=0, l2=1e-2))

        with tf.device("/cpu:0"):
            self(tf.constant(np.zeros(shape=(1,)+state_shape, dtype=np.float32)))

    def call(self, inputs):
        features = tf.nn.relu(self.l1(inputs))
        features = tf.nn.relu(self.l2(features))
        features = self.l3(features)
        action = tf.nn.tanh(features)
        action /= (tf.norm(action) + tf.keras.backend.epsilon())
        action *= self._action_space.high
        return action


class Critic(tf.keras.Model):
    def __init__(self, state_shape, action_dim, units=[400, 300], name="Critic"):
        super().__init__(name=name)

        self.l1 = Dense(units[0], name="L1", kernel_regularizer=regularizers.L1L2(l1=0, l2=1e-2))
        self.l2 = Dense(units[1], name="L2", kernel_regularizer=regularizers.L1L2(l1=0, l2=1e-2))
        self.l3 = Dense(1, name="L3", kernel_regularizer=regularizers.L1L2(l1=0, l2=1e-2))

        dummy_state = tf.constant(
            np.zeros(shape=(1,)+state_shape, dtype=np.float32))
        dummy_action = tf.constant(
            np.zeros(shape=[1, action_dim], dtype=np.float32))
        with tf.device("/cpu:0"):
            self([dummy_state, dummy_action])

    def call(self, inputs):
        states, actions = inputs
        features = tf.concat([states, actions], axis=1)
        features = tf.nn.relu(self.l1(features))
        features = tf.nn.relu(self.l2(features))
        features = self.l3(features)
        return features


class DDPG(OffPolicyAgent):
    def __init__(
            self,
            env,
            params,
            **kwargs):
        """Initializes a DDPG agent"""

        super().__init__(
            name=params["agent"]["name"],
            memory_capacity=params["agent"]["memory_capacity"],
            n_warmup=params["agent"]["n_warmup"],
            gpu=params["agent"]["gpu"],
            batch_size=params["agent"]["batch_size"],
            update_interval=params["agent"]["update_interval"],
            **kwargs
            )


        # Define and initialize Actor network
        self.actor = Actor(
            state_shape=env.observation_space.shape,
            action_space=env.action_space,
            units=params["agent"]["actor_units"]
            )
        self.actor_target = Actor(
            state_shape=env.observation_space.shape,
            action_space=env.action_space,
            units=params["agent"]["actor_units"]
            )
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=params["agent"]["lr_actor"])
        update_target_variables(self.actor_target.weights,
                                self.actor.weights, tau=1.)

        # Define and initialize Critic network
        self.critic = Critic(
            state_shape=env.observation_space.shape,
            action_dim=env.action_space.high.size,
            units=params["agent"]["critic_units"]
            )
        self.critic_target = Critic(
            state_shape=env.observation_space.shape,
            action_dim=env.action_space.high.size,
            units=params["agent"]["critic_units"]
            )
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=params["agent"]["lr_critic"])
        update_target_variables(
            self.critic_target.weights, self.critic.weights, tau=1.)

        # Set hyperparameters
        self.sigma = params["agent"]["sigma"]
        self.tau = params["agent"]["tau"]

        # in evaluation mode the action of the agent is deterministic, not stochastic.
        self.eval_mode = False


    def get_action(self, state, test=False, tensor=False):
        is_single_state = len(state.shape) == 1
        if not tensor:
            assert isinstance(state, np.ndarray)
        state = np.expand_dims(state, axis=0).astype(
            np.float32) if is_single_state else state
        action = self._get_action_body(tf.constant(state), self.sigma * (1. - test))
        if tensor:
            return action
        else:
            return action.numpy()[0] if is_single_state else action.numpy()

    @tf.function
    def _get_action_body(self, state, sigma):
        with tf.device(self.device):
            action = self.actor(state)
            if self.eval_mode:
                return action
            else:
                action += tf.random.normal(shape=action.shape,
                                        mean=0., stddev=sigma, dtype=tf.float32)
                return tf.clip_by_value(action, tf.constant(-1, dtype=tf.float32), tf.constant(1, dtype=tf.float32))

    def train(self, states, actions, next_states, rewards, done, weights=None):
        if weights is None:
            weights = np.ones_like(rewards)
        actor_loss, critic_loss, td_errors = self._train_body(
            states, actions, next_states, rewards, done, weights)

        if actor_loss is not None:
            tf.summary.scalar(name=self.policy_name+"/actor_loss",
                              data=actor_loss)
        tf.summary.scalar(name=self.policy_name+"/critic_loss",
                          data=critic_loss)

        return td_errors

    @tf.function
    def _train_body(self, states, actions, next_states, rewards, done, weights):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                td_errors = self._compute_td_error_body(
                    states, actions, next_states, rewards, done)
                critic_loss = tf.reduce_mean(
                    huber_loss(td_errors, delta=self.max_grad) * weights)

            critic_grad = tape.gradient(
                critic_loss, self.critic.trainable_variables)
            critic_grad = [(tf.clip_by_value(grad, tf.constant(-self.max_grad, dtype=tf.float32), 
                tf.constant(self.max_grad, dtype=tf.float32))) for grad in critic_grad]
            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic.trainable_variables))

            with tf.GradientTape() as tape:
                next_action = self.actor(states)
                actor_loss = -tf.reduce_mean(self.critic([states, next_action]))

            actor_grad = tape.gradient(
                actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor.trainable_variables))

            # Update target networks
            update_target_variables(
                self.critic_target.weights, self.critic.weights, self.tau)
            update_target_variables(
                self.actor_target.weights, self.actor.weights, self.tau)

            return actor_loss, critic_loss, td_errors

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        if isinstance(actions, tf.Tensor):
            rewards = tf.expand_dims(rewards, axis=1)
            dones = tf.expand_dims(dones, 1)
        td_errors = self._compute_td_error_body(
            states, actions, next_states, rewards, dones)
        return np.abs(np.ravel(td_errors.numpy()))

    @tf.function
    def _compute_td_error_body(self, states, actions, next_states, rewards, dones):
        with tf.device(self.device):
            not_dones = 1. - dones
            target_Q = self.critic_target(
                [next_states, self.actor_target(next_states)])
            target_Q = rewards + (not_dones * self.discount * target_Q)
            target_Q = tf.stop_gradient(target_Q)
            current_Q = self.critic([states, actions])
            td_errors = target_Q - current_Q
        return td_errors

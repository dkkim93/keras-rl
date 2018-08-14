from __future__ import division
import os
import warnings
import numpy as np
import keras.backend as K
import keras.optimizers as optimizers
from rl.core import Agent
from rl.util import *
from copy import deepcopy


def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))


# Deep DPG as described by Lillicrap et al. (2015)
# http://arxiv.org/pdf/1509.02971v2.pdf
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.646.4324&rep=rep1&type=pdf
class DDPGAgent(Agent):
    """Write me
    """
    def __init__(self, nb_actions, actor, critic, critic_action_input, memory,
                 gamma=.99, batch_size=32, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                 train_interval=1, memory_interval=1, delta_range=None, delta_clip=np.inf,
                 random_process=None, custom_model_objects={}, target_model_update=.001, policy_type=None, **kwargs):
        if hasattr(actor.output, '__len__') and len(actor.output) > 1:
            raise ValueError(
                'Actor "{}" has more than one output. \
                DDPG expects an actor that has a single output.'.format(actor))
        if hasattr(critic.output, '__len__') and len(critic.output) > 1:
            raise ValueError(
                'Critic "{}" has more than one output. \
                DDPG expects a critic that has a single output.'.format(critic))
        if critic_action_input not in critic.input:
            raise ValueError(
                'Critic "{}" does not have designated action input "{}".'.format(critic, critic_action_input))
        if not hasattr(critic.input, '__len__') or len(critic.input) < 2:
            raise ValueError(
                'Critic "{}" does not have enough inputs. \
                The critic must have at exactly two inputs, \
                one for the action and one for the observation.'.format(critic))

        super(DDPGAgent, self).__init__(**kwargs)

        # Soft vs hard target model updates.
        if target_model_update < 0:
            raise ValueError('`target_model_update` must be >= 0.')
        elif target_model_update >= 1:
            # Hard update every `target_model_update` steps.
            target_model_update = int(target_model_update)
        else:
            # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
            target_model_update = float(target_model_update)

        if delta_range is not None:
            warnings.warn(
                '`delta_range` is deprecated. Please use `delta_clip` instead, which takes a single scalar. \
                For now we\'re falling back to `delta_range[1] = {}`'.format(delta_range[1]))
            delta_clip = delta_range[1]

        # Parameters.
        self.nb_actions = nb_actions
        self.nb_steps_warmup_actor = nb_steps_warmup_actor
        self.nb_steps_warmup_critic = nb_steps_warmup_critic
        self.random_process = random_process
        self.delta_clip = delta_clip
        self.gamma = gamma
        self.target_model_update = target_model_update
        self.policy_type = policy_type
        self.batch_size = batch_size
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.custom_model_objects = custom_model_objects

        # Related objects.
        self.actor = actor
        self.critic = critic
        self.critic_action_input = critic_action_input
        self.critic_action_input_idx = self.critic.input.index(critic_action_input)
        self.memory = memory

        # State.
        self.compiled = False
        self.reset_states()

    @property
    def uses_learning_phase(self):
        return self.actor.uses_learning_phase or self.critic.uses_learning_phase

    def compile(self, optimizer, metrics=[], meta_n=None, i_policy=None):
        metrics += [mean_q]

        if type(optimizer) in (list, tuple):
            if len(optimizer) != 2:
                raise ValueError(
                    'More than two optimizers provided. Please only provide a maximum of two optimizers, \
                    the first one for the actor and the second one for the critic.')
            actor_optimizer, critic_optimizer = optimizer
        else:
            actor_optimizer = optimizer
            critic_optimizer = clone_optimizer(optimizer)
        if type(actor_optimizer) is str:
            actor_optimizer = optimizers.get(actor_optimizer)
        if type(critic_optimizer) is str:
            critic_optimizer = optimizers.get(critic_optimizer)
        assert actor_optimizer != critic_optimizer

        if len(metrics) == 2 and hasattr(metrics[0], '__len__') and hasattr(metrics[1], '__len__'):
            actor_metrics, critic_metrics = metrics
        else:
            actor_metrics = critic_metrics = metrics

        def clipped_error(y_true, y_pred):
            return K.mean(huber_loss(y_true, y_pred, self.delta_clip), axis=-1)

        # Compile target networks. We only use them in feed-forward mode, hence we can pass any
        # optimizer and loss since we never use it anyway.
        self.target_actor = clone_model(self.actor, self.custom_model_objects)
        self.target_actor.compile(optimizer='sgd', loss='mse')
        self.target_critic = clone_model(self.critic, self.custom_model_objects)
        self.target_critic.compile(optimizer='sgd', loss='mse')

        # We also compile the actor. We never optimize the actor using Keras but instead compute
        # the policy gradient ourselves. However, we need the actor in feed-forward mode, hence
        # we also compile it with any optimzer and
        self.actor.compile(optimizer='sgd', loss='mse')

        # Compile the critic.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            critic_updates = get_soft_target_model_updates(self.target_critic, self.critic, self.target_model_update)
            critic_optimizer = AdditionalUpdatesOptimizer(critic_optimizer, critic_updates)
        self.critic.compile(optimizer=critic_optimizer, loss=clipped_error, metrics=critic_metrics)

        # Combine actor and critic so that we can get the policy gradient.
        if self.policy_type == "exec":
            self.compile_exec(actor_optimizer)
        elif self.policy_type == "meta":
            self.compile_meta(actor_optimizer, meta_n, i_policy)
        else:
            raise ValueError()

    def compile_meta(self, actor_optimizer, meta_n, i_policy):
        # Set combined_inputs and output
        combined_inputs = []
        for i in self.critic.input:
            if i == self.critic_action_input:
                combined_inputs.append([])
            else:
                critic_obs_input = i
                combined_inputs.append(critic_obs_input)

        actor_input = []
        actor_input.append(self.actor.get_input_at(0))

        # We use placeholder for other meta policies actor output to make sure that 
        # gradient does not flow from this meta policy's critic to other meta policies' actors.
        actor_output = self.actor(actor_input)
        actor_output_other = K.placeholder(shape=(None, self.nb_actions * (len(meta_n) - 1)))  # Other agts actor output
        actor_output_n = K.concatenate([actor_output, actor_output_other], axis=-1)

        combined_inputs[self.critic_action_input_idx] = actor_output_n

        combined_output = self.critic(combined_inputs)

        # Create update
        updates = actor_optimizer.get_updates(
            params=self.actor.trainable_weights, 
            loss=-K.mean(combined_output))
        if self.target_model_update < 1.:
            # Include soft target model updates.
            updates += get_soft_target_model_updates(self.target_actor, self.actor, self.target_model_update)
        updates += self.actor.updates  # include other updates of the actor, e.g. for BN

        # Finally, combine it all into a callable function.
        if K.backend() == 'tensorflow':
            self.actor_train_fn = K.function(
                [actor_input[0], K.learning_phase(), actor_output_other, critic_obs_input], 
                [self.actor(actor_input)], 
                updates=updates)
        else:
            raise NotImplementedError()
            if self.uses_learning_phase:
                critic_inputs += [K.learning_phase()]
            self.actor_train_fn = K.function(critic_inputs, [self.actor(critic_inputs)], updates=updates)
        self.actor_optimizer = actor_optimizer

        self.compiled = True

    def compile_exec(self, actor_optimizer):
        combined_inputs = []
        critic_inputs = []
        for i in self.critic.input:
            if i == self.critic_action_input:
                combined_inputs.append([])
            else:
                combined_inputs.append(i)
                critic_inputs.append(i)

        combined_inputs[self.critic_action_input_idx] = self.actor(critic_inputs)

        combined_output = self.critic(combined_inputs)

        updates = actor_optimizer.get_updates(
            params=self.actor.trainable_weights, loss=-K.mean(combined_output))  # -sign to do gradient ascent
        if self.target_model_update < 1.:
            # Include soft target model updates.
            updates += get_soft_target_model_updates(self.target_actor, self.actor, self.target_model_update)
        updates += self.actor.updates  # include other updates of the actor, e.g. for BN

        # Finally, combine it all into a callable function.
        if K.backend() == 'tensorflow':
            self.actor_train_fn = K.function(
                critic_inputs + [K.learning_phase()],
                [self.actor(critic_inputs)], 
                updates=updates)
        else:
            if self.uses_learning_phase:
                critic_inputs += [K.learning_phase()]
            self.actor_train_fn = K.function(critic_inputs, [self.actor(critic_inputs)], updates=updates)
        self.actor_optimizer = actor_optimizer

        self.compiled = True

    def load_weights(self, filepath):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.load_weights(actor_filepath)
        self.critic.load_weights(critic_filepath)
        self.update_target_models_hard()

    def save_weights(self, filepath, overwrite=False):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.save_weights(actor_filepath, overwrite=overwrite)
        self.critic.save_weights(critic_filepath, overwrite=overwrite)

    def update_target_models_hard(self):
        self.target_critic.set_weights(self.critic.get_weights())
        self.target_actor.set_weights(self.actor.get_weights())

    # TODO: implement pickle

    def reset_states(self):
        if self.random_process is not None:
            self.random_process.reset_states()
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.actor.reset_states()
            self.critic.reset_states()
            self.target_actor.reset_states()
            self.target_critic.reset_states()

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def select_action(self, state):
        batch = self.process_state_batch([state])
        action = self.actor.predict_on_batch(batch).flatten()
        assert action.shape == (self.nb_actions,)

        # Apply noise, if a random process is set.
        if self.training and self.random_process is not None:
            noise = self.random_process.sample()
            assert noise.shape == action.shape
            action += noise

        return action

    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        action = self.select_action(state)  # TODO: move this into policy

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action

    @property
    def layers(self):
        return self.actor.layers[:] + self.critic.layers[:]

    @property
    def metrics_names(self):
        names = self.critic.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names

    def train_exec(self, total_step):
        experiences, _ = self.memory.sample(self.batch_size)
        assert len(experiences) == self.batch_size

        # Start by extracting the necessary parameters (we use a vectorized implementation).
        state0_batch = []
        reward_batch = []
        action_batch = []
        terminal1_batch = []
        state1_batch = []
        for e in experiences:
            state0_batch.append(e.state0)
            state1_batch.append(e.state1)
            reward_batch.append(e.reward)
            action_batch.append(e.action)
            terminal1_batch.append(0. if e.terminal1 else 1.)

        # Prepare and validate parameters.
        state0_batch = self.process_state_batch(state0_batch)
        state1_batch = self.process_state_batch(state1_batch)
        terminal1_batch = np.array(terminal1_batch)
        reward_batch = np.array(reward_batch)
        action_batch = np.array(action_batch)
        assert reward_batch.shape == (self.batch_size,)
        assert terminal1_batch.shape == reward_batch.shape
        assert action_batch.shape == (self.batch_size, self.nb_actions)

        # Update critic, if warm up is over.
        if total_step > self.nb_steps_warmup_critic:
            target_actions = self.target_actor.predict_on_batch(state1_batch)
            assert target_actions.shape == (self.batch_size, self.nb_actions)
            if len(self.critic.inputs) >= 3:
                state1_batch_with_action = state1_batch[:]
            else:
                state1_batch_with_action = [state1_batch]
            state1_batch_with_action.insert(self.critic_action_input_idx, target_actions)
            target_q_values = self.target_critic.predict_on_batch(state1_batch_with_action).flatten()
            assert target_q_values.shape == (self.batch_size,)

            # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
            # but only for the affected output units (as given by action_batch).
            discounted_reward_batch = self.gamma * target_q_values
            discounted_reward_batch *= terminal1_batch
            assert discounted_reward_batch.shape == reward_batch.shape
            targets = (reward_batch + discounted_reward_batch).reshape(self.batch_size, 1)

            # Perform a single batch update on the critic network.
            if len(self.critic.inputs) >= 3:
                state0_batch_with_action = state0_batch[:]
            else:
                state0_batch_with_action = [state0_batch]
            state0_batch_with_action.insert(self.critic_action_input_idx, action_batch)
            metrics = self.critic.train_on_batch(state0_batch_with_action, targets)
            if self.processor is not None:
                metrics += self.processor.metrics

        # Update actor, if warm up is over.
        if total_step > self.nb_steps_warmup_actor:
            # TODO: implement metrics for actor
            if len(self.actor.inputs) >= 2:
                inputs = state0_batch[:]
            else:
                inputs = [state0_batch]
            if self.uses_learning_phase:
                inputs += [self.training]
            action_values = self.actor_train_fn(inputs)[0]
            assert action_values.shape == (self.batch_size, self.nb_actions)

    def train_meta(self, total_step, meta_n, i_policy):
        assert meta_n is not None
        assert i_policy is not None

        experiences, batch_idxs = self.memory.sample(self.batch_size)
        assert len(experiences) == self.batch_size
        assert len(batch_idxs) == self.batch_size

        # Start by extracting the necessary parameters (we use a vectorized implementation).
        state0_batch = []
        reward_batch = []
        action_batch = []
        terminal1_batch = []
        state1_batch = []
        for e in experiences:
            state0_batch.append(e.state0)
            state1_batch.append(e.state1)
            reward_batch.append(e.reward)
            action_batch.append(e.action)
            terminal1_batch.append(0. if e.terminal1 else 1.)

        # Get state0_batch_n, state1_batch_n, action_batch_n 
        # to train centralized critic
        state0_batch_n = deepcopy(state0_batch)
        state1_batch_n = deepcopy(state1_batch)
        action_batch_n = deepcopy(action_batch)
        meta_order_n = [i_policy]
        for i_meta in range(len(meta_n)):
            if i_meta != i_policy:
                # NOTE -1 as there is +1 in the sampling function
                experiences_other_agt, _ = \
                    meta_n[i_meta].policy.memory.sample(self.batch_size, np.array(batch_idxs) - 1)
                meta_order_n.append(i_meta)

                for i_exp, exp in enumerate(experiences_other_agt):
                    assert reward_batch[i_exp] == exp.reward

                    state0_batch_n[i_exp][0] = np.concatenate((
                        state0_batch_n[i_exp][0], exp.state0[0]))
                    state1_batch_n[i_exp][0] = np.concatenate((
                        state1_batch_n[i_exp][0], exp.state1[0]))
                    action_batch_n[i_exp] = np.concatenate((
                        action_batch_n[i_exp], exp.action))

        # Prepare and validate parameters.
        state0_batch = self.process_state_batch(state0_batch)
        state1_batch = self.process_state_batch(state1_batch)
        terminal1_batch = np.array(terminal1_batch)
        reward_batch = np.array(reward_batch)
        action_batch = np.array(action_batch)

        state0_batch_n = self.process_state_batch(state0_batch_n)
        state1_batch_n = self.process_state_batch(state1_batch_n)
        action_batch_n = np.array(action_batch_n)
        assert reward_batch.shape == (self.batch_size,)
        assert terminal1_batch.shape == reward_batch.shape
        assert action_batch.shape == (self.batch_size, self.nb_actions)
        assert action_batch_n.shape == (self.batch_size, self.nb_actions * len(meta_n))

        # Update critic, if warm up is over.
        if total_step > self.nb_steps_warmup_critic:
            # Get target_action_n
            # Because this is centralized critic, we need target_action for all agents
            # TODO Need more carefull thought for agt > 2
            target_actions = self.target_actor.predict_on_batch(state1_batch)

            target_actions_n = deepcopy(target_actions)
            for i_meta in range(len(meta_n)):
                if i_meta > 0:
                    interval = state0_batch.shape[-1]
                    next_state_batch = state1_batch_n[:, :, i_meta * interval:(i_meta + 1) * interval]
                    target_actions = meta_n[meta_order_n[i_meta]].policy.target_actor.predict_on_batch(next_state_batch)

                    target_actions_n = np.concatenate((target_actions_n, target_actions), axis=1)

                    assert next_state_batch.shape == (self.batch_size, 1, interval)
                    assert target_actions.shape == (self.batch_size, self.nb_actions)
            assert target_actions_n.shape == (self.batch_size, self.nb_actions * len(meta_n))

            if len(self.critic.inputs) >= 3:
                state1_batch_n_with_action = state1_batch_n[:]
            else:
                state1_batch_n_with_action = [state1_batch_n]
            state1_batch_n_with_action.insert(
                self.critic_action_input_idx, target_actions_n)  # Put target_actions in front by input idx

            target_q_values = self.target_critic.predict_on_batch(state1_batch_n_with_action).flatten()
            assert target_q_values.shape == (self.batch_size,)

            # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
            # but only for the affected output units (as given by action_batch).
            discounted_reward_batch = self.gamma * target_q_values
            discounted_reward_batch *= terminal1_batch
            assert discounted_reward_batch.shape == reward_batch.shape
            targets = (reward_batch + discounted_reward_batch).reshape(self.batch_size, 1)

            # Perform a single batch update on the critic network.
            if len(self.critic.inputs) >= 3:
                state0_batch_n_with_action = state0_batch_n[:]
            else:
                state0_batch_n_with_action = [state0_batch_n]
            state0_batch_n_with_action.insert(self.critic_action_input_idx, action_batch_n)
            metrics = self.critic.train_on_batch(state0_batch_n_with_action, targets)

            if self.processor is not None:
                metrics += self.processor.metrics

        # Update actor, if warm up is over.
        if total_step > self.nb_steps_warmup_actor:
            # TODO: implement metrics for actor
            if len(self.actor.inputs) >= 2:
                inputs = state0_batch[:]
            else:
                inputs = [state0_batch]
            
            if self.uses_learning_phase:
                inputs += [self.training]

            actor_output_other = action_batch_n[:, 2:]
            critic_obs_input = state0_batch_n
            action_values = self.actor_train_fn(
                [inputs[0], self.training, actor_output_other, critic_obs_input])[0]
            assert action_values.shape == (self.batch_size, self.nb_actions)

    def add_memory(self, reward, total_step, terminal=False):
        # Store most recent experience in memory.
        if total_step % self.memory_interval == 0:
            self.memory.append(
                self.recent_observation, self.recent_action, reward, terminal, training=self.training)

    def backward(self, total_step, meta_n=None, i_policy=None):
        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Check batch size, if not enough then no training
        # Times 2 to make sure we have enough batch :-)
        if len(self.memory.rewards) < self.batch_size * 2:
            return metrics

        # Train the network on a single stochastic batch.
        can_train_either = total_step > self.nb_steps_warmup_critic or total_step > self.nb_steps_warmup_actor
        if can_train_either and total_step % self.train_interval == 0:
            if self.policy_type == "exec":
                self.train_exec(total_step)
            elif self.policy_type == "meta":
                # Check whether all memories have same length
                memory_len = meta_n[0].policy.memory.nb_entries
                for i_meta in range(len(meta_n)):
                    if i_meta != i_policy:
                        memory_len_diff = abs(meta_n[i_meta].policy.memory.nb_entries - memory_len)
                        assert memory_len_diff == 0

                self.train_meta(total_step, meta_n, i_policy)
            else:
                raise ValueError()

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_models_hard()

        return metrics

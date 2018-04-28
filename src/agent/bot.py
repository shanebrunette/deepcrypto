import tensorflow as tf
import numpy as np
import random
import collections
import datetime
import sys
import pandas as pd

import net
from memory import RingMemory, EpisodeMemory


class BotBase():

    def __init__(self, gym, env_name, args):
        self.env_name = env_name
        self.env = self.get_env(gym, env_name, args)
        self.set_hyper_parameters(args)
        self.online_net = self.get_net(scope='online')
        self.target_net = self.get_net(scope='target')
        self.size_action = self.online_net.size_action
        self.sess = self.start_session()
        self.writer = self.start_writer()
        self.saver = tf.train.Saver() 
        self._update_ops = self.update_target_graph()
        self.memory = self.get_memory()
        self.render = args.render if args.gym else False
        self.shuffle = args.shuffle if args.coinlist else False
        self.verbose = args.verbose
        self.start_balance = 1000
        self._record = []

    def set_hyper_parameters(self, args):
        self.gamma = args.gamma 
        self.epsilon = args.epsilon
        self.target_epsilon = args.target_epsilon 
        self.epsilon_strength = args.epsilon_strength
        self.memory_size = args.memory_size
        self.batch_size = args.batch_size
        self.episodes = args.episodes
        self.ep_max_steps = args.ep_max_steps
        self.hidden_nodes = args.hidden_nodes
        self.bias = args.bias
        self.tau = args.tau
        self.trace_size = args.trace_size
        self.dropout = args.dropout

    def get_env(self, gym, env_name, args):
        env = gym.make(env_name)
        if args.seed and args.gym:
            env.seed(args.seed)
        return env

    def start_session(self):
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        return sess

    def start_writer(self):
        logdir = "tensorboard/" + datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S") + "/"
        writer = tf.summary.FileWriter(logdir, self.sess.graph)
        return writer

    def q_train(self):
        self.batch_presentations = 0
        episode_rewards = [self.train_ep() for ep in range(self.episodes)]
        return episode_rewards 

    def train_ep(self):     
        self.update_exploration_rate()
        ep_reward = self.step_through_env()
        if self.verbose:
            self.print_record(ep_reward)
        return ep_reward

    def update_exploration_rate(self):
        exploration_remaining = self.epsilon - self.target_epsilon
        decay_rate = exploration_remaining / self.epsilon_strength 
        self.epsilon -= decay_rate

    def reset_env(self):
        if self.shuffle: 
            self.env = self.env.make(self.env_name)
        state = self.env.reset()
        if self.render: 
            self.env.render()
        return state

    def train_network(self):
        if self.memory.size > self.batch_size:
            summary = self.train_on_memory()
            self.record(summary)
            self.update_target_network()

    def record(self, summary):
        self.batch_presentations += 1
        if self.batch_presentations % 10 == 0:
            self.writer.add_summary(summary, self.batch_presentations)
        if self.batch_presentations % 2000 == 0:
            self.saver.save(self.sess, "/tmp/model.ckpt")

    def update_target_graph(self):
        all_vars = tf.trainable_variables()
        split = len(all_vars)//2
        online_vars = all_vars[:split]
        target_vars = all_vars[split:]
        operations = []
        for i in range(split):
            operations.append(target_vars[i].assign(
                    online_vars[i].value() * self.tau
                    + target_vars[i].value() * (1-self.tau)
            ))
        return operations

    def update_target_network(self):
        for op in self._update_ops:
            self.sess.run(op)

    def print_record(self, ep_reward):
        self._record.append(ep_reward)
        print(f"""
            count: {len(self._record)},
            running ave reward: {np.mean(self._record[-100:])},
            ep reward: {ep_reward},
            epsilon: {self.epsilon}
            """
        )


class BotLSTM(BotBase):

    def get_net(self, scope):
        return net.NetLSTM(
            env=self.env,
            scope=scope,
            hidden_nodes=self.hidden_nodes, 
            bias=self.bias,
            dropout=self.dropout
        )

    def get_memory(self):
        return EpisodeMemory(self.memory_size)

    def step_through_env(self):

        def explore_episode():
            s,a,r,s1,f = 0,1,2,3,4
            ep_reward = 0
            episode_events = []
            state = self.reset_env()
            rnn_state = self.online_net.get_start_rnn_state(1)
            for step in range(self.ep_max_steps):
                event, rnn_state = take_action(state, rnn_state)
                state, reward, finished = event[s1], event[r], event[f]
                ep_reward += reward
                episode_events.append(event)
                self.train_network()
                if finished:
                    break
            memorize(episode_events)
            return ep_reward

        def take_action(state, rnn_state):
            action, rnn_state = get_action(state, rnn_state)
            next_state, reward, finished, _ = self.env.step(action)
            if self.render: 
                self.env.render()
            event = np.array([state, action, reward, next_state, finished])
            return event, rnn_state

        def memorize(episode):
            if len(episode) > self.trace_size:
                episode = np.array(episode)
                self.memory.append(episode)
            else:
                print('warning: episode discarded')

        def get_action(state, rnn_state):
            if random.random() < self.epsilon:
                action = [random.randint(0, self.size_action - 1)]
            else:
                state_input = np.array(state)[np.newaxis]
                action, rnn_state = self.sess.run([
                        self.online_net.predict_action,
                        self.online_net.rnn_state, 
                    ], feed_dict={
                        self.online_net.state_input: state_input,
                        self.online_net.batch_size: 1,
                        self.online_net.rnn_state0: rnn_state[0],
                        self.online_net.rnn_state1: rnn_state[1],
                    })
            return action[0], rnn_state
        
        episode_reward = explore_episode()
        return episode_reward

    def train_on_memory(self):

        def train():
            target, state, action = get_train_batch()
            rnn_blank_state = self.online_net.blank_rnn(self.batch_size)
            ops = [self.online_net.train_summary, self.online_net.optimize_step]
            summary, _ = self.sess.run(ops, feed_dict={
                    self.online_net.target_q_values: target,
                    self.online_net.state_input: np.vstack(state),
                    self.online_net.action_input: action,
                    self.online_net.batch_size: self.batch_size,
                    self.online_net.rnn_state0: rnn_blank_state,
                    self.online_net.rnn_state1: rnn_blank_state,
                })
            return summary

        def get_train_batch():
            s,a,r,s1,f = 0,1,2,3,4
            remembered = self.memory.sample(self.batch_size, self.trace_size)
            next_q = get_q_values(remembered[:,s1])
            mask = remembered[:,f] == 0
            target = remembered[:,r] + (self.gamma * next_q * mask)
            return target, remembered[:,s], remembered[:,a]

        def get_q_values(state):
            state = np.vstack(state)
            action = self.online_net.predict_action.eval(
                                feed_dict=feed_dict(self.online_net, state))
            q_values = self.target_net.q_values.eval(
                                feed_dict=feed_dict(self.target_net, state))
            return q_values[range(len(action)), action]

        def feed_dict(net, state):
            return {
                net.state_input: state,
                net.batch_size: self.batch_size,
                net.rnn_state0: net.blank_rnn(self.batch_size),
                net.rnn_state1: net.blank_rnn(self.batch_size)
            }

        summary = train()
        return summary


class BotFF(BotBase):

    def get_net(self, scope):
        return net.Net(
            env=self.env,
            scope=scope,
            hidden_nodes=self.hidden_nodes, 
            bias=self.bias,
            dropout=self.dropout
        )

    def get_memory(self):
        return RingMemory(self.memory_size)

    def step_through_env(self):

        def explore_episode():
            s,a,r,s1,f = 0,1,2,3,4
            ep_reward = 0
            state = self.reset_env()
            for step in range(self.ep_max_steps):
                event = take_action(state)
                state, reward, finished = event[s1], event[r], event[f]
                ep_reward += reward
                memorize(event)
                self.train_network()
                if finished:
                    break
            return ep_reward

        def take_action(state):
            action = get_action(state)
            next_state, reward, finished, _ = self.env.step(action)
            if self.render: 
                self.env.render()
            event = np.array([state, action, reward, next_state, finished])
            return event

        def memorize(event):
            self.memory.append(np.array(event))

        def get_action(state):
            if random.random() < self.epsilon:
                action = random.randint(0, self.size_action - 1)
            else:
                state_input = np.array(state)[np.newaxis]
                action = self.sess.run([
                        self.online_net.predict_action
                    ], feed_dict={
                        self.online_net.state_input: state_input,
                        self.online_net.batch_size: 1,
                    })[0][0]
            return action
        
        episode_reward = explore_episode()
        return episode_reward

    def train_on_memory(self):

        def train():
            target, state, action = get_train_batch()
            ops = [self.online_net.train_summary, self.online_net.optimize_step]
            summary, _ = self.sess.run(ops, feed_dict={
                    self.online_net.target_q_values: target,
                    self.online_net.state_input: np.vstack(state),
                    self.online_net.action_input: action,
                    self.online_net.batch_size: self.batch_size,
                })
            return summary

        def get_train_batch():
            s,a,r,s1,f = 0,1,2,3,4
            remembered = self.memory.sample(self.batch_size)
            next_q = get_q_values(remembered[:,s1])
            mask = remembered[:,f] == 0
            target = remembered[:,r] + (self.gamma * next_q * mask)
            return target, remembered[:,s], remembered[:,a]

        def get_q_values(state):
            state = np.vstack(state)
            action = self.online_net.predict_action.eval(
                                feed_dict=feed_dict(self.online_net, state))
            q_values = self.target_net.q_values.eval(
                                feed_dict=feed_dict(self.target_net, state))
            return q_values[range(len(action)), action]

        def feed_dict(net, state):
            return {
                net.state_input: state,
                net.batch_size: self.batch_size,
            }

        summary = train()
        return summary


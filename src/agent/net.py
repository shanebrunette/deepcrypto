import tensorflow as tf
import numpy as np


class Net:

    def __init__(self, env, scope, hidden_nodes=5, bias=0.1, dropout=0.0):
        self.scope = scope
        self.hidden_nodes = hidden_nodes
        self.bias = bias
        self.dropout = dropout
        self.size_state, self.size_action = self.get_size(env)
        self.get_placeholders()
        self.q_values = self.get_output()
        self.predict_action = self.get_predicted_action()
        self.action_q_values = self.get_action_q()
        self.loss = self.get_loss()
        self.optimize_step = self.get_optimizer()
        self.train_summary = self.get_summary()

    def get_size(self, env):
        return env.observation_space.shape[0], env.action_space.n

    def get_placeholders(self):
        with tf.variable_scope(f'placeholders_{self.scope}'):
            self.batch_size = tf.placeholder_with_default(1,
                                    shape=(),
                                    name='batch_size')
            self.state_input = tf.placeholder(
                                    dtype=tf.float32, 
                                    shape=[None, self.size_state], 
                                    name='state_input')
            self.action_input = tf.placeholder(
                                    dtype=tf.int32,
                                    shape=[None],
                                    name='action_input')
            self.target_q_values = tf.placeholder(
                                    dtype=tf.float32, 
                                    shape=[None],
                                    name='target_q_values')

    def get_output(self):
        hidden_output = self.feed_forward_stack(self.state_input)       
        q_values = self.duelling_stream(hidden_output)
        return q_values

    def feed_forward_stack(self, state_input):
        with tf.variable_scope(f'FF_Stack_{self.scope}'):
            first_layer = self.feed_forward(
                    input_data=state_input,
                    input_shape=self.size_state,
                    output_shape=self.hidden_nodes,
                    name='first_feed_forward',
                    relu=True
            )
            second_layer = self.feed_forward(
                    input_data=first_layer,
                    input_shape=self.hidden_nodes,
                    output_shape=self.hidden_nodes,
                    name='second_feed_forward',
                    relu=True
            )
            return second_layer

    def feed_forward(self, 
                     input_data, 
                     input_shape, 
                     output_shape, 
                     name, 
                     relu=False):
        weights = self.get_weights(
                input_shape=input_shape, 
                output_shape=output_shape,
                name=name
        )
        biases = self.get_biases(shape=output_shape, name=name)
        layer = tf.matmul(input_data, weights) + biases
        if relu:
            return tf.nn.relu(layer)
        else:
            return layer

    def get_weights(self, input_shape, output_shape, name):
        return tf.get_variable(
            shape=[input_shape, output_shape],
            initializer=tf.contrib.layers.xavier_initializer(),
            dtype=tf.float32,
            name=f'{name}_weights_{self.scope}'
        ) 


    def get_biases(self, shape, name):
        return tf.get_variable(
            initializer=tf.constant(self.bias, shape=[shape]),
            dtype=tf.float32,
            name=f'{name}_biases_{self.scope}'
        ) 

    def duelling_stream(self, output_layer):
        # https://arxiv.org/pdf/1511.06581.pdf
        with tf.name_scope(f'value_stream_{self.scope}'):
            value = self.stream_feed_forward_layer(
                            input_data=output_layer,
                            shape=1, 
                            name='value')
        with tf.name_scope(f'advantage_stream_{self.scope}'):
            advantage = self.stream_feed_forward_layer(
                            input_data=output_layer,
                            shape=self.size_action, 
                            name='advantage')
        with tf.name_scope(f'q_value_calc_{self.scope}'):
            q_values = self.calculate_advantage(value, advantage)
        return q_values

    def stream_feed_forward_layer(self, input_data, shape, name):
        return self.feed_forward(
            input_data=input_data,
            input_shape=self.hidden_nodes,
            output_shape=shape,
            name=name,
            relu=False
        )

    def calculate_advantage(self, value, advantage):
        # https://arxiv.org/pdf/1511.06581.pdf e.q. 9
        advantage_mean = tf.reduce_mean(advantage, axis=1, keep_dims=True)
        return value + tf.subtract(advantage, advantage_mean)


    def get_predicted_action(self):
        with tf.name_scope(f'best_action_{self.scope}'):
            return tf.argmax(self.q_values, axis=1)


    def get_action_q(self):
        one_hot_action = self.get_one_hot_action()
        masked_action_score = tf.multiply(self.q_values, one_hot_action)
        action_score = tf.reduce_sum(masked_action_score, axis=1)
        return action_score

    def get_one_hot_action(self):
        return tf.one_hot(
                self.action_input, 
                depth=self.size_action, 
                dtype=tf.float32
        )

    def get_loss(self):
        prediction_error = self.target_q_values - self.action_q_values
        return tf.reduce_mean(tf.square(prediction_error))


    def get_optimizer(self):
        optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        return optimizer


    def get_summary(self):
        summary = tf.summary.scalar('TrainingLoss', self.loss)
        return summary


class NetLSTM(Net):
      
    def get_output(self):
        feed_forward_layer = self.feed_forward_stack(self.state_input)  
        memory_layer = self.recurrent_layer(feed_forward_layer)
        q_values = self.duelling_stream(memory_layer)
        return q_values

    def recurrent_layer(self, input_data):
        # https://arxiv.org/pdf/1507.06527.pdf - lstm and dql
        with tf.name_scope(f'LSTM_{self.scope}'):
            shaped_data = tf.reshape(
                            input_data, 
                            [self.batch_size, -1, self.hidden_nodes])
            rnn_cell = self.rnn_cell()
            rnn_state = self.rnn_state()
            output, self.rnn_state = tf.nn.dynamic_rnn(
                                            inputs=shaped_data,
                                            cell=rnn_cell,
                                            dtype=tf.float32,
                                            initial_state=rnn_state,
                                            scope=f'lstm_unit_{self.scope}')
            return tf.reshape(output, [-1, self.hidden_nodes])

    def rnn_cell(self):
        # https://arxiv.org/pdf/1409.2329.pdf - dropout and rnn
        with tf.variable_scope(f'lstm_cell_{self.scope}'):
            lstm = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_nodes,
                                                    state_is_tuple=True)
            lstm_with_dropout = tf.contrib.rnn.DropoutWrapper(
                                        cell=lstm, 
                                        output_keep_prob=(1-self.dropout))
            return lstm

    def rnn_state(self):        
        self.rnn_state0 = self.lstm_state_placeholder('state0')
        self.rnn_state1 = self.lstm_state_placeholder('state1')
        lstm_state = tf.contrib.rnn.LSTMStateTuple(
                                        self.rnn_state0, 
                                        self.rnn_state1)
        return lstm_state

    def lstm_state_placeholder(self, name):
        return tf.placeholder(
            dtype=tf.float32, 
            shape=[None, self.hidden_nodes],
            name=f'lstm_{name}_{self.scope}'
        )


    def get_start_rnn_state(self, batch_size):
        return (self.blank_rnn(batch_size),
                                             self.blank_rnn(batch_size))

    def blank_rnn(self, batch_size):
        return np.zeros([batch_size, self.hidden_nodes])


if __name__ == '__main__':
    import gym
    env = gym.make('CartPole-v0')
    net = NetLSTM(env, scope='test')
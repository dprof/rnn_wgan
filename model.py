import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import GRUCell



from config import *


def Discriminator_GRU(inputs, charmap_len, seq_len, reuse=False):
    with tf.variable_scope("Discriminator", reuse=reuse):
        num_neurons = FLAGS.DISC_STATE_SIZE

        weight = tf.get_variable("embedding", shape=[charmap_len, num_neurons],
                                 initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

        # backwards compatability
        if FLAGS.DISC_GRU_LAYERS == 1:
            cell = GRUCell(num_neurons)
        else:
            cell = tf.contrib.rnn.MultiRNNCell([GRUCell(num_neurons) for _ in range(FLAGS.DISC_GRU_LAYERS)], state_is_tuple=True)

        flat_inputs = tf.reshape(inputs, [-1, charmap_len])

        inputs = tf.reshape(tf.matmul(flat_inputs, weight), [-1, seq_len, num_neurons])
        inputs = tf.unstack(tf.transpose(inputs, [1,0,2]))


        for inp in inputs:
            print(inp.get_shape())

        #Q: Why not a call to dynamic RNN?
        output, state = tf.contrib.rnn.static_rnn(
            cell,
            inputs,
            dtype=tf.float32
        )
        #Output shape is (64,512)

        last = output[-1]

        weight2 = tf.get_variable("W", shape=[num_neurons, 1],
                                 initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
        bias = tf.get_variable("b", shape=[1], initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

        prediction = tf.matmul(last, weight2) + bias

        return prediction # (64,1)

#Note Discriminator_LSTM closely follows Discriminator_GRU - difference is in the state handling for LSTM cell
def Discriminator_LSTM(inputs, charmap_len, seq_len, reuse=False):
    with tf.variable_scope("Discriminator", reuse=reuse):
        num_neurons = FLAGS.DISC_STATE_SIZE

        weight = tf.get_variable("embedding", shape=[charmap_len, num_neurons],
                                 initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

        # backwards compatability
        if FLAGS.DISC_GRU_LAYERS == 1:
            cell = GRUCell(num_neurons)
        else:
            cell = tf.contrib.rnn.MultiRNNCell([GRUCell(num_neurons) for _ in range(FLAGS.DISC_GRU_LAYERS)], state_is_tuple=True)

        flat_inputs = tf.reshape(inputs, [-1, charmap_len])

        inputs = tf.reshape(tf.matmul(flat_inputs, weight), [-1, seq_len, num_neurons])
        inputs = tf.unstack(tf.transpose(inputs, [1,0,2]))


        for inp in inputs:
            print(inp.get_shape())

        #Q: Why not a call to dynamic RNN?
        output, state = tf.contrib.rnn.static_rnn(
            cell,
            inputs,
            dtype=tf.float32
        )
        #Output shape is (64,512)

        last = output[-1]

        weight2 = tf.get_variable("W", shape=[num_neurons, 1],
                                 initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
        bias = tf.get_variable("b", shape=[1], initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

        prediction = tf.matmul(last, weight2) + bias

        return prediction # (64,1)


def Generator_GRU_CL_VL_TH(n_samples, charmap_len, seq_len=None, gt=None):
    with tf.variable_scope("Generator"):
        noise, noise_shape = get_noise()
        #print("noise, noise_shape", noise, noise_shape)
        num_neurons = FLAGS.GEN_STATE_SIZE

        cells = []
        for l in range(FLAGS.GEN_GRU_LAYERS):
            #from tensorflow.contrib.rnn
            cells.append(GRUCell(num_neurons))
            #cells.append(LSTMCell(num_neurons))

        # this is separate to decouple train and test
        train_initial_states = create_initial_states_gru(noise)
        inference_initial_states = create_initial_states_gru(noise)

        sm_weight = tf.Variable(tf.random_uniform([num_neurons, charmap_len], minval=-0.1, maxval=0.1))
        sm_bias = tf.Variable(tf.random_uniform([charmap_len], minval=-0.1, maxval=0.1))

        embedding = tf.Variable(tf.random_uniform([charmap_len, num_neurons], minval=-0.1, maxval=0.1))

        char_input = tf.Variable(tf.random_uniform([num_neurons], minval=-0.1, maxval=0.1))
        char_input = tf.reshape(tf.tile(char_input, [n_samples]), [n_samples, 1, num_neurons])

        if seq_len is None:
            seq_len = tf.placeholder(tf.int32, None, name="ground_truth_sequence_length")

        if gt is not None: #if no GT, we are training
            #print("Generator - Training")
            train_pred = get_train_op(cells, char_input, charmap_len, embedding, gt, n_samples, num_neurons, seq_len,
                                      sm_bias, sm_weight, train_initial_states)
            #Teacher Helping happens here
            inference_op = get_inference_op(cells, char_input, embedding, seq_len, sm_bias, sm_weight, inference_initial_states,
                                            num_neurons,
                                            charmap_len, reuse=True)
        else:
            #Inferencing
            inference_op = get_inference_op(cells, char_input, embedding, seq_len, sm_bias, sm_weight, inference_initial_states,
                                            num_neurons,
                                            charmap_len, reuse=False)
            train_pred = None

        return train_pred, inference_op

#Note Generator_LSTM_CL_VL_TH closely follows Generator_GRU_CL_VL_TH - difference is in the
#state handling for LSTM cell
def Generator_LSTM_CL_VL_TH(n_samples, charmap_len, seq_len=None, gt=None):
    with tf.variable_scope("Generator"):
        noise, noise_shape = get_noise()
        #print("noise, noise_shape", noise, noise_shape)
        num_neurons = FLAGS.GEN_STATE_SIZE

        cells = []
        for l in range(FLAGS.GEN_GRU_LAYERS):
            #from tensorflow.contrib.rnn
            cells.append(LSTMCell(num_neurons))

        # this is separate to decouple train and test
        train_initial_hidden_states = create_initial_states_lstm(noise) #[list of Dim (BSIZE, STATESIZE)]
        train_initial_current_states = create_initial_states_lstm(noise)  # list of Dim (BSIZE, STATESIZE)
        train_initial_states = (train_initial_hidden_states, train_initial_current_states)
        inference_initial_hidden_states = create_initial_states_lstm(noise)  # list of Dim (BSIZE, STATESIZE)
        inference_initial_current_states = create_initial_states_lstm(noise)  # list of Dim (BSIZE, STATESIZE)
        inference_initial_states = (inference_initial_hidden_states, inference_initial_current_states)

        sm_weight = tf.Variable(tf.random_uniform([num_neurons, charmap_len], minval=-0.1, maxval=0.1))
        sm_bias = tf.Variable(tf.random_uniform([charmap_len], minval=-0.1, maxval=0.1))

        embedding = tf.Variable(tf.random_uniform([charmap_len, num_neurons], minval=-0.1, maxval=0.1))

        char_input = tf.Variable(tf.random_uniform([num_neurons], minval=-0.1, maxval=0.1))
        char_input = tf.reshape(tf.tile(char_input, [n_samples]), [n_samples, 1, num_neurons])

        if seq_len is None:
            seq_len = tf.placeholder(tf.int32, None, name="ground_truth_sequence_length")

        if gt is not None: #if no GT, we are training
            #print("Generator - Training")
            train_pred = get_train_op_lstm(cells, char_input, charmap_len, embedding, gt, n_samples, num_neurons, seq_len,
                                      sm_bias, sm_weight, train_initial_states)
            #Teacher Helping happens here
            inference_op = get_inference_op_lstm(cells, char_input, embedding, seq_len, sm_bias, sm_weight, inference_initial_states,
                                            num_neurons,
                                            charmap_len, reuse=True)
        else:
            #Inferencing
            inference_op = get_inference_op_lstm(cells, char_input, embedding, seq_len, sm_bias, sm_weight, inference_initial_states,
                                            num_neurons,
                                            charmap_len, reuse=False)
            train_pred = None

        return train_pred, inference_op


def create_initial_states_gru(noise):
    states = []
    for l in range(FLAGS.GEN_GRU_LAYERS):
        states.append(noise)
    return states

def create_initial_states_lstm(noise):
    states = []
    for l in range(FLAGS.GEN_LSTM_LAYERS):
        states.append(noise)
    return states

def get_train_op(cells, char_input, charmap_len, embedding, gt, n_samples, num_neurons, seq_len, sm_bias, sm_weight,
                 states):
    gt_embedding = tf.reshape(gt, [n_samples * seq_len, charmap_len])
    gt_GRU_input = tf.matmul(gt_embedding, embedding)
    gt_GRU_input = tf.reshape(gt_GRU_input, [n_samples, seq_len, num_neurons])[:, :-1]

    gt_sentence_input = tf.concat([char_input, gt_GRU_input], axis=1)

    GRU_output, _ = rnn_step_prediction(cells, charmap_len, gt_sentence_input, num_neurons, seq_len, sm_bias,
                                         sm_weight,
                                         states)
    train_pred = []
    # TODO: optimize loop
    for i in range(seq_len):
        train_pred.append(
            tf.concat([tf.zeros([BATCH_SIZE, seq_len - i - 1, charmap_len]), gt[:, :i], GRU_output[:, i:i + 1, :]],
                      axis=1))

    train_pred = tf.reshape(train_pred, [BATCH_SIZE*seq_len, seq_len, charmap_len])

    if FLAGS.LIMIT_BATCH:
        #What's happening here???
        indices = tf.random_uniform([BATCH_SIZE], 0, BATCH_SIZE*seq_len, dtype=tf.int32)
        train_pred = tf.gather(train_pred, indices)


    return train_pred

#To fix
def get_train_op_lstm(cells, char_input, charmap_len, embedding, gt, n_samples, num_neurons, seq_len, sm_bias, sm_weight,
                 states):
    gt_embedding = tf.reshape(gt, [n_samples * seq_len, charmap_len])
    gt_LSTM_input = tf.matmul(gt_embedding, embedding)
    gt_LSTM_input = tf.reshape(gt_LSTM_input, [n_samples, seq_len, num_neurons])[:, :-1]

    gt_sentence_input = tf.concat([char_input, gt_LSTM_input], axis=1)

    LSTM_output, _ = rnn_step_prediction_lstm(cells, charmap_len, gt_sentence_input, num_neurons, seq_len, sm_bias,
                                         sm_weight,
                                         states)
    train_pred = []
    # TODO: optimize loop
    for i in range(seq_len):
        train_pred.append(
            tf.concat([tf.zeros([BATCH_SIZE, seq_len - i - 1, charmap_len]), gt[:, :i], LSTM_output[:, i:i + 1, :]],
                      axis=1))

    train_pred = tf.reshape(train_pred, [BATCH_SIZE*seq_len, seq_len, charmap_len])

    if FLAGS.LIMIT_BATCH:
        #What's happening here???
        indices = tf.random_uniform([BATCH_SIZE], 0, BATCH_SIZE*seq_len, dtype=tf.int32)
        train_pred = tf.gather(train_pred, indices)


    return train_pred

def rnn_step_prediction(cells, charmap_len, gt_sentence_input, num_neurons, seq_len, sm_bias, sm_weight, states,
                        reuse=False):
    with tf.variable_scope("rnn", reuse=reuse):
        GRU_output = gt_sentence_input

        #states is a list of l elements, each of dim (BSIZE, STATESIZE)
        for l in range(FLAGS.GEN_GRU_LAYERS):
            #Addition of LSTM needs to happen here using tf.nn.dynamic_rnn ...
            #print("l=", l)
            GRU_output, states[l] = tf.nn.dynamic_rnn(cells[l], GRU_output, dtype=tf.float32,
                                                       initial_state=states[l], scope="layer_%d" % (l + 1))

    GRU_output = tf.reshape(GRU_output, [-1, num_neurons])
    GRU_output = tf.nn.softmax(tf.matmul(GRU_output, sm_weight) + sm_bias)
    GRU_output = tf.reshape(GRU_output, [BATCH_SIZE, -1, charmap_len])

    return GRU_output, states

#To fix
def rnn_step_prediction_lstm(cells, charmap_len, gt_sentence_input, num_neurons, seq_len, sm_bias, sm_weight, states,
                        reuse=False):
    with tf.variable_scope("rnn", reuse=reuse):
        LSTM_output = gt_sentence_input

        for l in range(FLAGS.GEN_LSTM_LAYERS):
            #Addition of LSTM needs to happen here using tf.nn.dynamic_rnn ...
            #print("l=", l)
            #print("layer_%d" % (l + 1))
            state_tuple = tf.nn.rnn_cell.LSTMStateTuple(states[0][l], states[1][l])
            LSTM_output, state_tuple = tf.nn.dynamic_rnn(cells[l], LSTM_output, dtype=tf.float32,
                                            initial_state=state_tuple, scope="layer_%d" % (l + 1))

    LSTM_output = tf.reshape(LSTM_output, [-1, num_neurons])
    LSTM_output = tf.nn.softmax(tf.matmul(LSTM_output, sm_weight) + sm_bias)
    LSTM_output = tf.reshape(LSTM_output, [BATCH_SIZE, -1, charmap_len])

    return LSTM_output, states

def get_inference_op(cells, char_input, embedding, seq_len, sm_bias, sm_weight, states, num_neurons, charmap_len,
                     reuse=False):
    inference_pred = []
    embedded_pred = [char_input]
    for i in range(seq_len):

        step_pred, states = rnn_step_prediction(cells, charmap_len, tf.concat(embedded_pred, 1), num_neurons, seq_len,
                                                sm_bias, sm_weight, states, reuse=reuse)
        best_chars_tensor = tf.argmax(step_pred, axis=2)
        best_chars_one_hot_tensor = tf.one_hot(best_chars_tensor, charmap_len)
        best_char = best_chars_one_hot_tensor[:, -1, :]
        inference_pred.append(tf.expand_dims(best_char, 1))
        embedded_pred.append(tf.expand_dims(tf.matmul(best_char, embedding), 1))
        reuse = True  # no matter what the reuse was, after the first step we have to reuse the defined vars

    result_inf_pred = tf.concat(inference_pred, axis=1)
    return result_inf_pred

#To fix
def get_inference_op_lstm(cells, char_input, embedding, seq_len, sm_bias, sm_weight, states, num_neurons, charmap_len,
                     reuse=False):
    inference_pred = []
    embedded_pred = [char_input]
    for i in range(seq_len):

        step_pred, states = rnn_step_prediction_lstm(cells, charmap_len, tf.concat(embedded_pred, 1), num_neurons, seq_len,
                                                sm_bias, sm_weight, states, reuse=reuse)
        best_chars_tensor = tf.argmax(step_pred, axis=2)
        best_chars_one_hot_tensor = tf.one_hot(best_chars_tensor, charmap_len)
        best_char = best_chars_one_hot_tensor[:, -1, :]
        inference_pred.append(tf.expand_dims(best_char, 1))
        embedded_pred.append(tf.expand_dims(tf.matmul(best_char, embedding), 1))
        reuse = True  # no matter what the reuse was, after the first step we have to reuse the defined vars

    result_inf_pred = tf.concat(inference_pred, axis=1)
    return result_inf_pred

generators = {
    "Generator_GRU_CL_VL_TH": Generator_GRU_CL_VL_TH,
    "Generator_LSTM_CL_VL_TH": Generator_LSTM_CL_VL_TH,
}

discriminators = {
    "Discriminator_GRU": Discriminator_GRU,
    "Discriminator_LSTM": Discriminator_LSTM,
}


def get_noise():
    noise_shape = [BATCH_SIZE, FLAGS.GEN_STATE_SIZE]
    return make_noise(shape=noise_shape, stddev=FLAGS.NOISE_STDEV), noise_shape


def get_generator(model_name):
    return generators[model_name]


def params_with_name(name):
    return [p for p in tf.trainable_variables() if name in p.name]


def get_discriminator(model_name):
    return discriminators[model_name]


def make_noise(shape, mean=0.0, stddev=1.0):
    return tf.random_normal(shape, mean, stddev)
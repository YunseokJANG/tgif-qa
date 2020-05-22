import tensorflow as tf
from tensorflow.python.client import device_lib
from model_saver import ModelSaver
from util import log
import tensorflow.contrib.rnn as rnn
from rnn_cell.custom_rnn_cell import LayerNormBasicLSTMCell, MultiRNNCell
from ops import *

import time
import numpy as np

class CountBase(ModelSaver):

    PARAMS = [
        "feat_dim",
        "hidden_dim",
        "batch_size",
        "lstm_steps",
        "word_embed",
        "num_layers",
        "name",
        "dropout_keep_prob",
        "architecture",
        "att_hidden_dim",
    ]

    def __init__(self,
                 hidden_dim,
                 lstm_steps,
                 word_embed,
                 feat_dim=[1, 1, 2048],
                 batch_size=100,
                 num_layers=2,
                 name="Count",
                 dropout_keep_prob=0.8,
                 vocabulary_size=12000,
                 word_dim=300,
                 architecture="1video2text",
                 att_hidden_dim=512):


        self.name = name
        self.word_embed = word_embed
        if word_embed is not None:
            self.vocabulary_size = self.word_embed.shape[0]
            self.word_dim = self.word_embed.shape[1]
        else:
            self.vocabulary_size = vocabulary_size
            self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.att_hidden_dim = att_hidden_dim
        self.lstm_steps = lstm_steps
        self.feat_dim = feat_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout_keep_prob = dropout_keep_prob


        self.feat_dims_arr = self.feat_dim
        self.kernel_size = self.feat_dims_arr[0]
        self.channel_size = self.feat_dims_arr[2]

        self.N_PRETRAIN = 3000
        self.step = 0
        self.train_step = tf.placeholder(tf.int32)

        self.devices = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
        self.batch_size_per_gpu = batch_size/len(self.devices)
        self.mean_loss_list = []
        self.mean_grad_list = []
        self.eval_loss_list = []
        self.alpha_list = []
        self.predictions_list = []
        self.correct_predictions_list = []
        self.architecture = architecture
        self.acc_list = []


    @staticmethod
    def add_flags():
        pass

    def get_feed_dict(self, batch_chunk):
        feed_dict = {
            self.video: batch_chunk['video_features'].astype(float),
            self.video_mask: batch_chunk['video_mask'].astype(float),
            self.caption: batch_chunk['question_words'],
            self.caption_mask: batch_chunk['question_mask'],
            self.answer: batch_chunk['answer'],
            self.train_step: self.step,
        }
        return feed_dict

    def build_graph(self,
                    video,
                    video_mask,
                    question,
                    question_mask,
                    answer,
                    optimizer):


        self.video = video  # [batch_size, length, kernel, kernel, channel]
        self.video_mask = video_mask  # [batch_size, length]
        self.caption = question
        self.caption_mask = question_mask  # [batch_size, length]
        self.answer = answer
        self.optimizer = optimizer


        # word embedding and dropout, etc.
        if self.word_embed is not None:
            self.word_embed_t = tf.constant(self.word_embed, dtype=tf.float32, name="word_embed")
        else:
            self.word_embed_t = tf.get_variable("Word_embed",
                                                [self.vocabulary_size, self.word_dim],
                                                initializer=tf.random_normal_initializer(stddev=0.1))
        self.dropout_keep_prob_t = tf.placeholder_with_default(1., [])


        with tf.variable_scope("conv_image_emb"):
            self.r_shape = tf.reshape(self.video, [-1, self.kernel_size, self.kernel_size, self.channel_size])
            #  [batch_size*length, kernel_size, kernel_size, channel_size]
            self.pooled_feat = tf.nn.avg_pool(self.r_shape,
                                              ksize=[1, self.kernel_size, self.kernel_size, 1],
                                              strides=[1, self.kernel_size, self.kernel_size, 1],
                                              padding="SAME")
            #  [batch_size*length, 1, 1, channel_size]
            self.squeezed_feat = tf.squeeze(self.pooled_feat)
            #  [batch_size*length, channel_size]
            self.embedded_feat = tf.reshape(self.squeezed_feat, [self.batch_size,
                                                                 self.lstm_steps,
                                                                 self.channel_size])
            #  [batch_size, length, channel_size]
            self.embedded_feat_drop = tf.nn.dropout(self.embedded_feat, self.dropout_keep_prob_t)


        with tf.variable_scope("word_emb"):
            with tf.device("/cpu:0"):
                self.embedded_captions = tf.nn.embedding_lookup(self.word_embed_t, self.caption)
                # [batch_size, length, word_dim]
                self.embedded_start_word = tf.nn.embedding_lookup(self.word_embed_t,
                                                                  tf.ones([self.batch_size], dtype=tf.int32))

        if self.architecture == "1video2text":
            vid_rnn_states = self.video_rnn()
            cap_rnn_states = self.cap_rnn(vid_rnn_states)
            rnn_states = cap_rnn_states

        elif self.architecture == "1text2video":
            cap_rnn_states = self.cap_rnn()
            vid_rnn_states = self.video_rnn(cap_rnn_states)
            rnn_states = vid_rnn_states

        elif self.architecture == "parallel":
            # both start with 0 states,
            cap_rnn_states = self.cap_rnn()
            vid_rnn_states = self.video_rnn(cap_rnn_states)
            rnn_states = cap_rnn_states

        with tf.variable_scope("loss") as scope:
            rnn_final_state = tf.concat([rnn_states[-1][0][0], rnn_states[-1][1][0]], 1)
            if self.architecture == "parallel":
                cap_final_state = rnn_final_state
                vid_final_state = tf.concat([vid_rnn_states[-1][0][0], vid_rnn_states[-1][1][0]], 1)
                # concat the two states and add it to vid_att
                final_embed = linear(tf.concat([rnn_final_state, vid_final_state], 1), 2*self.hidden_dim, name="final_state_mapping")
            else:
                final_embed = rnn_final_state

            rnnW = tf.get_variable(
                "W",
                [2*self.hidden_dim, 1],
                initializer=tf.random_normal_initializer(stddev=0.1))
            rnnb = tf.get_variable(
                "b",
                [1],
                initializer=tf.constant_initializer(0.0))
            logits = tf.nn.xw_plus_b(final_embed,rnnW,rnnb)
            self.logits = logits

        # prediction tensor on test phase
        self.predictions = tf.cast(tf.clip_by_value(tf.round(logits), 1, 10), tf.int64)

        self.mean_loss = tf.reduce_mean(tf.square(tf.subtract(
                tf.cast(self.logits, tf.float32), tf.cast(self.answer, tf.float32))))

        self.eval_loss = tf.reduce_mean(tf.square(tf.subtract(
                tf.cast(self.predictions, tf.float32), tf.cast(self.answer, tf.float32))))

        with tf.variable_scope("gradient") as scope:
            gs, vs = zip(*self.optimizer.compute_gradients(self.mean_loss))
            clipped_gs, _ = tf.clip_by_global_norm(gs, clip_norm=5)
            self.mean_grad_list.append(zip(clipped_gs, vs))
            # from IPython import embed; embed()
            self.mean_grad = average_gradients(self.mean_grad_list) # use this to debug.

        with tf.variable_scope("accuracy"):
            self.correct_predictions = tf.cast(tf.equal(
                tf.reshape(self.predictions, [self.batch_size, 1]),
                tf.cast(self.answer,tf.int64)), tf.int32)
            self.acc = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

    def get_rnn_cell(self):
        return rnn.DropoutWrapper(
            LayerNormBasicLSTMCell(self.hidden_dim),
            input_keep_prob=self.dropout_keep_prob_t,
            output_keep_prob=self.dropout_keep_prob_t)

    def video_rnn(self, cap_rnn_states=None):
        with tf.variable_scope("video_rnn") as scope:
            self.video_cell = MultiRNNCell([self.get_rnn_cell()] * (self.num_layers))

            if self.architecture == "1video2text" or self.architecture == "parallel":
                vid_rnn_states = [self.video_cell.zero_state(self.batch_size, tf.float32)]
            elif self.architecture == "1text2video":
                vid_rnn_states = [cap_rnn_states[-1]]

            for i in range(self.lstm_steps):
                if i > 0:
                    scope.reuse_variables()

                new_output, new_state = self.video_cell(
                    self.embedded_feat_drop[:, i, :], vid_rnn_states[-1]
                )

                vid_rnn_states.append(
                    (
                    (new_state[0][0]*tf.expand_dims(self.video_mask[:, i], 1),
                     new_state[0][1]*tf.expand_dims(self.video_mask[:, i], 1)),
                    (new_state[1][0]*tf.expand_dims(self.video_mask[:, i], 1),
                     new_state[1][1]*tf.expand_dims(self.video_mask[:, i], 1))
                    )
                )

        return vid_rnn_states

    def cap_rnn(self, vid_rnn_states=None):
        with tf.variable_scope("caption_rnn") as scope:
            self.caption_cell = MultiRNNCell([self.get_rnn_cell()] *self.num_layers)
            if self.architecture == "1video2text":
                cap_rnn_states = [vid_rnn_states[-1]]
            elif self.architecture == "1text2video" or self.architecture == "parallel":
                # set cap-rnn_states to zeros.
                cap_rnn_states = [self.caption_cell.zero_state(self.batch_size, tf.float32)]

            current_embedded_y = self.embedded_start_word
            for i in range(self.lstm_steps):
                if i > 0:
                    scope.reuse_variables()

                new_output, new_state = self.caption_cell(
                    current_embedded_y, cap_rnn_states[-1]
                )
                cap_rnn_states.append(new_state)
                current_embedded_y = self.embedded_captions[:, i, :]

        return cap_rnn_states


class CountBaseEvaluator:
    def __init__(self, model, summary_dir=None):
        self.model = model
        with tf.variable_scope("evaluation"):
            self.summary_writer = None
            if summary_dir is not None:
                self.summary_writer = tf.summary.FileWriter(summary_dir)
            self.build_eval_graph()

    def build_eval_graph(self):
        # Keep track of the totals while running through the batch data
        self.total_loss = tf.Variable(0.0, trainable=False, collections=[])
        self.total_correct = tf.Variable(0, trainable=False, collections=[])
        self.total_l2 = tf.Variable(0., trainable=False, collections=[])
        self.example_count = tf.Variable(0, trainable=False, collections=[])
        example_count_as_float = tf.cast(self.example_count, 'float32')

        # Calculates the means
        self.mean_loss = self.total_loss * self.model.batch_size / example_count_as_float
        self.accuracy = tf.cast(self.total_correct, 'float32') / example_count_as_float
        self.eval_loss = tf.cast(self.total_l2, 'float32') * self.model.batch_size / example_count_as_float

        # Operations to modify to the stateful variables
        inc_total_loss = self.total_loss.assign_add(self.model.mean_loss)
        inc_total_l2 = self.total_l2.assign_add(self.model.eval_loss)
        inc_total_correct = self.total_correct.assign_add(
            tf.reduce_sum(self.model.correct_predictions))
        inc_example_count = self.example_count.assign_add(self.model.batch_size)

        with tf.control_dependencies([self.total_loss.initializer,
                                      self.total_l2.initializer,
                                      self.total_correct.initializer,
                                      self.example_count.initializer]):
            self.eval_reset = tf.no_op(name='eval_reset')

        with tf.control_dependencies([inc_total_loss, inc_total_l2, inc_total_correct, inc_example_count]):
            self.eval_step = tf.no_op(name='eval_step')

        self.summary_v_loss = tf.summary.scalar("v_loss", self.mean_loss)
        self.summary_v_acc = tf.summary.scalar("v_acc", self.accuracy)
        self.summary_v_l2 = tf.summary.scalar("v_acc", self.eval_loss)


    def eval(self, batch_iter, test_size, global_step=None, sess=None,
             generate_results=False):

        sess = sess or tf.get_default_session()
        global_step = global_step or tf.no_op()
        sess.run(self.eval_reset)

        result_json = []

        for k, batch_chunk in enumerate(batch_iter):

            feed_dict = self.model.get_feed_dict(batch_chunk)

            pred, val_acc, eval_loss_, _, mask = sess.run(
                [self.model.predictions, self.model.acc, self.model.eval_loss,
                 self.eval_step, self.model.video_mask], feed_dict=feed_dict)

            vid_len = mask.sum(axis=1, dtype=np.int32).tolist()
            pred = pred.reshape(-1)

            if k % 5 == 0:
                current_accuracy, current_examples = sess.run(
                    [self.accuracy, self.example_count])
                log.infov('Evaluation step %d, current accuracy = %.3f (%d), acc = %.3f, eval_loss = %.3f',
                          k, current_accuracy, current_examples, val_acc, eval_loss_)

            # SAMPLING
            if generate_results:
                for j, pred_j in enumerate(pred):
                    cor = 0
                    if pred_j == batch_chunk['answer'][j]:
                        cor = 1
                    result_json.append({
                        'id' : batch_chunk['ids'][j],
                        'pred' : int(pred_j),
                        'ans' : int(batch_chunk['answer'][j]),
                        'question' : batch_chunk['debug_sent'][j],
                        'correct' : cor,
                        'vid_length' : vid_len[j]
                    })

        loss, acc, eval_loss, sumstr_vloss, sumstr_vacc, sumstr_vl2, current_step = \
            sess.run([self.mean_loss, self.accuracy, self.eval_loss,
                      self.summary_v_loss, self.summary_v_acc, self.summary_v_l2, global_step])
        if self.summary_writer is not None:
            self.summary_writer.add_summary(sumstr_vloss, current_step)
            self.summary_writer.add_summary(sumstr_vacc, current_step)
            self.summary_writer.add_summary(sumstr_vl2, current_step)

        # Adjust loss from duplicated data
        N = (k+1) * self.model.batch_size
        if N > test_size:
            pred_ = pred[:N-test_size]
            ans_ = batch_chunk['answer'][:N-test_size].reshape(-1)
            eval_loss = eval_loss*N - eval_loss_*self.model.batch_size  + ((pred_-ans_)**2).sum()
            eval_loss /= test_size

        # SAMPLING
        if generate_results:
            result_json_dict = {}
            for item in result_json:
                result_json_dict[item['id']] = item
            result_json = []
            for k in sorted(result_json_dict.keys()):
                result_json.append(result_json_dict[k])

        return [eval_loss, acc, current_step, result_json]


class CountBaseTrainer:

    def __init__(self, model, optimizer=None, train_summary_dir=None, sess=None, max_grad_norm=5):
        sess = sess or tf.get_default_session()
        self.model = model
        with tf.variable_scope("training", reuse=None):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.optimizer = optimizer or tf.train.AdadeltaOptimizer()

            self.train_op = self.optimizer.apply_gradients(
                model.mean_grad, global_step=self.global_step)

            self.summary_mean_loss = tf.summary.scalar("mean_loss", model.mean_loss)
            self.train_summary_writer = None
            if train_summary_dir is not None:
                self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph_def)

    def train_loop(self, train_iter, sess=None):
        sess = sess or tf.get_default_session()
        for batch_chunk in train_iter:
            start_ts = time.time()
            feed_dict = self.model.get_feed_dict(batch_chunk)
            feed_dict[self.model.dropout_keep_prob_t] = self.model.dropout_keep_prob

            _, train_loss, train_acc, train_l2, current_step, summary = sess.run(
                [self.train_op, self.model.mean_loss, self.model.acc, self.model.eval_loss, self.global_step, self.summary_mean_loss],
                feed_dict=feed_dict)

            if self.train_summary_writer is not None:
                self.train_summary_writer.add_summary(summary, current_step)

            end_ts = time.time()
            self.model.step += 1
            yield train_l2, train_acc, current_step, (end_ts - start_ts)

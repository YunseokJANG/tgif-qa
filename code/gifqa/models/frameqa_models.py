import tensorflow as tf
from tensorflow.python.client import device_lib
from frameqa_base import *
from util import log
from ops import *

import time
import numpy as np

class FrameQAResnet(FrameQABase):
    @staticmethod
    def add_flags(FLAGS):
        FLAGS.image_feature_net = "resnet"
        FLAGS.layer = "pool5"
class FrameQAResnetEvaluator(FrameQABaseEvaluator):
    pass
class FrameQAResnetTrainer(FrameQABaseTrainer):
    pass

class FrameQAC3D(FrameQABase):
    @staticmethod
    def add_flags(FLAGS):
        FLAGS.image_feature_net = "c3d"
        FLAGS.layer = "fc6"
class FrameQAC3DEvaluator(FrameQABaseEvaluator):
    pass
class FrameQAC3DTrainer(FrameQABaseTrainer):
    pass

class FrameQAOF(FrameQABase):
    @staticmethod
    def add_flags(FLAGS):
        FLAGS.image_feature_net = "optflow"
        FLAGS.layer = "pool5"
class FrameQAOFEvaluator(FrameQABaseEvaluator):
    pass
class FrameQAOFTrainer(FrameQABaseTrainer):
    pass

class FrameQAConcat(FrameQABase):
    @staticmethod
    def add_flags(FLAGS):
        FLAGS.image_feature_net = "concat"
        if FLAGS.feature == "C3D":
            FLAGS.layer = "fc"
        elif FLAGS.feature == "optflow":
            FLAGS.layer = "pool5"
class FrameQAConcatEvaluator(FrameQABaseEvaluator):
    pass
class FrameQAConcatTrainer(FrameQABaseTrainer):
    pass

class FrameQATp(FrameQABase):
    @staticmethod
    def add_flags(FLAGS):
        if FLAGS.feature == "optflow":
            FLAGS.image_feature_net = "concat"
            # FLAGS.image_feature_net = "resnet"
            FLAGS.layer = "pool5"
        elif FLAGS.feature == "C3D":
            FLAGS.image_feature_net = "concat"
            FLAGS.layer = "fc"

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
            self.vid_rnn_states, self.vid_states = self.video_rnn()
            self.cap_rnn_states = self.cap_rnn(self.vid_rnn_states)
            self.rnn_states = self.cap_rnn_states

        elif self.architecture == "1text2video":
            self.cap_rnn_states = self.cap_rnn()
            self.vid_rnn_states, self.vid_states = self.video_rnn(self.cap_rnn_states)
            self.rnn_states = self.cap_rnn_states # NOTE: vid_rnn_states here don't train.

        elif self.architecture == "parallel":
            # both start with 0 states,
            self.cap_rnn_states = self.cap_rnn()
            self.vid_rnn_states, self.vid_states = self.video_rnn(self.cap_rnn_states)
            self.rnn_states = self.cap_rnn_states

        with tf.variable_scope("merge") as scope:
            rnn_final_state = tf.concat([self.rnn_states[-1][0][0], self.rnn_states[-1][1][0]], 1)

            if self.architecture == "parallel":
                cap_final_state = rnn_final_state
                vid_final_state = tf.concat([self.vid_rnn_states[-1][0][0], self.vid_rnn_states[-1][1][0]], 1)
                # concat the two states and add it to vid_att
                rnn_final_state = linear(tf.concat([rnn_final_state, vid_final_state], 1), 2*self.hidden_dim, name="final_state_mapping")

            vid_att, alpha = self.attention(rnn_final_state, self.vid_states)
            self.alpha = alpha
            final_embed = tf.add(tf.nn.tanh(linear(vid_att, 2*self.hidden_dim)), rnn_final_state)

        with tf.variable_scope("loss") as scope:
            rnnW = tf.get_variable(
                "W",
                [2*self.hidden_dim, self.answer_size],
                initializer=tf.random_normal_initializer(stddev=0.1))
            rnnb = tf.get_variable(
                "b",
                [self.answer_size],
                initializer=tf.constant_initializer(0.0))
            embed_state = tf.nn.xw_plus_b(final_embed, rnnW,rnnb)

            labels = self.answer
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
            labels_with_index = tf.concat([indices, labels], 1)

            onehot_labels = tf.sparse_to_dense(labels_with_index,
                                                tf.stack([self.batch_size, self.answer_size]),
                                                sparse_values=1.0,
                                                default_value=0)
            cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=embed_state, labels=onehot_labels)

            self.mean_loss = tf.reduce_mean(cross_entropy_loss, name="t_loss")

        with tf.variable_scope("gradient") as scope:
            gs, vs = zip(*self.optimizer.compute_gradients(self.mean_loss))
            clipped_gs, _ = tf.clip_by_global_norm(gs, clip_norm=5)
            self.mean_grad_list.append(zip(clipped_gs, vs))
            self.mean_grad = average_gradients(self.mean_grad_list) # use this to debug.

        with tf.variable_scope("accuracy"):
            # prediction tensor on test phase
            self.predictions = tf.argmax(
                tf.reshape(embed_state, [self.batch_size, self.answer_size]),
                dimension=1, name='argmax_predictions'
            )
            self.predictions.get_shape().assert_is_compatible_with([self.batch_size])

            self.correct_predictions = tf.cast(tf.equal(
                tf.reshape(self.predictions, [self.batch_size, 1]),
                tf.cast(self.answer,tf.int64)), tf.int32)
            self.acc = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

    def attention(self, prev_hidden, vid_states):
        packed = tf.stack(vid_states)
        packed = tf.transpose(packed, [1,0,2])
        vid_2d = tf.reshape(packed, [-1, self.hidden_dim*2])
        sent_2d = tf.tile(prev_hidden, [1, self.lstm_steps])
        sent_2d = tf.reshape(sent_2d, [-1, self.hidden_dim*2])
        preact = tf.add(linear(sent_2d, self.att_hidden_dim, name="preatt_sent"),
                        linear(vid_2d, self.att_hidden_dim, name="preadd_vid"))
        score = linear(tf.nn.tanh(preact), 1, name="preatt")
        score_2d = tf.reshape(score, [-1, self.lstm_steps])
        alpha = tf.nn.softmax(score_2d)
        alpha_3d = tf.reshape(alpha, [-1, self.lstm_steps, 1])
        return tf.reduce_sum(packed * alpha_3d, 1), alpha


    def video_rnn(self, cap_rnn_states=None):
        with tf.variable_scope("video_rnn") as scope:
            self.video_cell = MultiRNNCell([self.get_rnn_cell()] * self.num_layers)

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
        vid_states = [
            tf.concat([vid_rnn_state[0][0], vid_rnn_state[1][0]], 1)
            for vid_rnn_state in vid_rnn_states[1:]
        ]

        return vid_rnn_states, vid_states

    def cap_rnn(self, vid_rnn_states=None):
        with tf.variable_scope("caption_rnn") as scope:
            self.caption_cell = MultiRNNCell([self.get_rnn_cell()] * self.num_layers)

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


class FrameQATpEvaluator(FrameQABaseEvaluator):
    pass
class FrameQATpTrainer(FrameQABaseTrainer):
    pass

class FrameQASp(FrameQABase):
    @staticmethod
    def add_flags(FLAGS):
        if FLAGS.feature == "optflow":
            FLAGS.image_feature_net = "concat"
            FLAGS.layer = "res5c"
        elif FLAGS.feature == "C3D":
            FLAGS.image_feature_net = "concat"
            FLAGS.layer = "conv"

    def build_graph(self,
                    video,
                    video_mask,
                    caption,
                    caption_mask,
                    answer,
                    optimizer):

        self.video = video  # [batch_size, length, kernel, kernel, channel]
        self.video_mask = video_mask  # [batch_size, length]
        self.caption = caption  # [batch_size, 5, length]
        self.caption_mask = caption_mask  # [batch_size, 5, length]
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

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for idx, device in enumerate(self.devices):
                with tf.device("/%s" % device):
                    if idx > 0:
                        tf.get_variable_scope().reuse_variables()
                    from_idx = self.batch_size_per_gpu*idx

                    video = tf.slice(self.video, [from_idx,0,0,0,0],
                                        [self.batch_size_per_gpu,-1,-1,-1,-1])
                    video_mask = tf.slice(self.video_mask, [from_idx,0],
                                            [self.batch_size_per_gpu,-1])
                    caption = tf.slice(self.caption, [from_idx,0],
                                            [self.batch_size_per_gpu,-1])
                    caption_mask = tf.slice(self.caption_mask, [from_idx,0],
                                                [self.batch_size_per_gpu,-1])
                    answer = tf.slice(self.answer, [from_idx,0],
                                                [self.batch_size_per_gpu,-1])

                    self.build_graph_single_gpu(video, video_mask, caption,
                                                caption_mask, answer, idx)

        self.mean_loss = tf.reduce_mean(tf.stack(self.mean_loss_list, axis=0))
        self.mean_grad = average_gradients(self.mean_grad_list)
        self.alpha = tf.stack(self.alpha_list, axis=0)
        self.predictions = tf.stack(self.predictions_list, axis=0)
        self.correct_predictions = tf.stack(self.correct_predictions_list, axis=0)
        self.acc = tf.reduce_mean(tf.stack(self.acc_list, axis=0))

    def build_graph_single_gpu(self, video, video_mask, caption, caption_mask, answer, idx):

        with tf.variable_scope("word_emb"):
            with tf.device("/cpu:0"):
                embedded_captions = tf.nn.embedding_lookup(self.word_embed_t, caption)
                # [batch_size, length, word_dim]
                embedded_start_word = tf.nn.embedding_lookup(
                    self.word_embed_t, tf.ones([self.batch_size_per_gpu], dtype=tf.int32))

        with tf.variable_scope("caption_rnn") as scope:
            caption_cell = MultiRNNCell([self.get_rnn_cell()] * self.num_layers)
            cap_rnn_states = [caption_cell.zero_state(self.batch_size_per_gpu, tf.float32)]
            current_embedded_y = embedded_start_word
            for i in range(self.lstm_steps):
                if i > 0:
                    scope.reuse_variables()

                new_output, new_state = caption_cell(current_embedded_y, cap_rnn_states[-1])
                cap_rnn_states.append(new_state)
                current_embedded_y = embedded_captions[:, i, :]

        with tf.variable_scope("merge_emb") as scope:
            rnn_final_state1 = tf.concat([cap_rnn_states[-1][0][1], cap_rnn_states[-1][1][1]], 1)
            fc_sent = linear(rnn_final_state1, 512, name="fc_sent")
            fc_sent = tf.tile(fc_sent, [1, self.lstm_steps*7*7])
            fc_sent = tf.reshape(fc_sent, [-1, 512])

            video_2d = tf.reshape(video, [self.batch_size_per_gpu*self.lstm_steps*7*7,
                                            self.channel_size])
            fc_vid = linear(video_2d, 512, name="fc_vid")
            pooled = tf.tanh(tf.add(fc_vid, fc_sent))

        with tf.variable_scope("att_image_emb"):
            pre_alpha = linear(pooled, 1, name="pre_alpha")
            pre_alpha = tf.reshape(pre_alpha, [-1, 7*7])
            alpha = tf.nn.softmax(pre_alpha)
            alpha = tf.reshape(alpha, [self.batch_size_per_gpu*self.lstm_steps, 7*7, 1])
            self.alpha_list.append(
                tf.reshape(alpha, [self.batch_size_per_gpu, self.lstm_steps, 7*7]))

            batch_pre_att = tf.reshape(video, [self.batch_size_per_gpu*self.lstm_steps,
                                                7*7, self.channel_size])
            embedded_feat = tf.reduce_sum(batch_pre_att * alpha, 1)
            embedded_feat = tf.reshape(embedded_feat, [self.batch_size_per_gpu, self.lstm_steps, self.channel_size])

            #  [batch_size, length, channel_size]
            embedded_feat_drop = tf.nn.dropout(
                embedded_feat, self.dropout_keep_prob_t)

        with tf.variable_scope("video_rnn") as scope:
            video_cell = MultiRNNCell([self.get_rnn_cell()] * self.num_layers)
            vid_rnn_states = [video_cell.zero_state(self.batch_size_per_gpu, tf.float32)]

            for i in range(self.lstm_steps):
                if i > 0:
                    scope.reuse_variables()
                new_output, new_state = video_cell(
                    embedded_feat_drop[:, i, :], vid_rnn_states[-1]
                )
                vid_rnn_states.append(
                    (
                        (new_state[0][0]*tf.expand_dims(video_mask[:, i], 1),
                        new_state[0][1]*tf.expand_dims(video_mask[:, i], 1)),
                        (new_state[1][0]*tf.expand_dims(video_mask[:, i], 1),
                        new_state[1][1]*tf.expand_dims(video_mask[:, i], 1))
                    )
                )

            vid_states = [
                tf.concat([vid_rnn_state[0][0], vid_rnn_state[1][0]], 1)
                for vid_rnn_state in vid_rnn_states[1:]
            ]

        with tf.variable_scope("caption_rnn") as scope:
            scope.reuse_variables()
            caption_cell = MultiRNNCell([self.get_rnn_cell()] * self.num_layers)
            cap_rnn_states = [vid_rnn_states[-1]]
            current_embedded_y = embedded_start_word
            for i in range(self.lstm_steps):
                if i > 0:
                    scope.reuse_variables()

                new_output, new_state = caption_cell(current_embedded_y, cap_rnn_states[-1])
                cap_rnn_states.append(new_state)
                current_embedded_y = embedded_captions[:, i, :]

        with tf.variable_scope("loss") as scope:
            rnn_final_state = tf.concat([cap_rnn_states[-1][0][0], cap_rnn_states[-1][1][0]], 1)
            rnnW = tf.get_variable(
                "W",
                [2*self.hidden_dim, self.answer_size],
                initializer=tf.random_normal_initializer(stddev=0.1))
            rnnb = tf.get_variable(
                "b",
                [self.answer_size],
                initializer=tf.constant_initializer(0.0))
            embed_state = tf.nn.xw_plus_b(rnn_final_state,rnnW,rnnb)

            labels = answer
            indices = tf.expand_dims(tf.range(0, self.batch_size_per_gpu, 1), 1)
            labels_with_index = tf.concat([indices, labels], 1)

            onehot_labels = tf.sparse_to_dense(labels_with_index,
                                                tf.stack([self.batch_size_per_gpu, self.answer_size]),
                                                sparse_values=1.0,
                                                default_value=0)
            cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=embed_state, labels=onehot_labels)

            mean_loss = tf.reduce_mean(cross_entropy_loss, name="t_loss")
            self.mean_loss_list.append(mean_loss)

        with tf.variable_scope("gradient") as scope:
            gs, vs = zip(*self.optimizer.compute_gradients(mean_loss))
            clipped_gs, _ = tf.clip_by_global_norm(gs, clip_norm=5)
            self.mean_grad_list.append(zip(clipped_gs, vs))

        with tf.variable_scope("accuracy"):
            # prediction tensor on test phase
            predictions = tf.argmax(
                tf.reshape(embed_state, [self.batch_size_per_gpu, self.answer_size]),
                dimension=1, name='argmax_predictions'
            )
            predictions.get_shape().assert_is_compatible_with([self.batch_size_per_gpu])

            correct_predictions = tf.cast(tf.equal(
                tf.reshape(predictions, [self.batch_size_per_gpu, 1]),
                tf.cast(answer,tf.int64)), tf.int32)
            acc = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy_%d"%idx)

            self.predictions_list.append(predictions)
            self.correct_predictions_list.append(correct_predictions)
            self.acc_list.append(acc)

class FrameQASpEvaluator(FrameQABaseEvaluator):
    pass
class FrameQASpTrainer(FrameQABaseTrainer):
    pass

class FrameQASpTp(FrameQABase):
    @staticmethod
    def add_flags(FLAGS):
        if FLAGS.feature == "optflow":
            FLAGS.image_feature_net = "concat"
            FLAGS.layer = "res5c"
        elif FLAGS.feature == "C3D":
            FLAGS.image_feature_net = "concat"
            FLAGS.layer = "conv"

    def build_graph(self,
                    video,
                    video_mask,
                    caption,
                    caption_mask,
                    answer,
                    optimizer):

        self.video = video  # [batch_size, length, kernel, kernel, channel]
        self.video_mask = video_mask  # [batch_size, length]
        self.caption = caption  # [batch_size, 5, length]
        self.caption_mask = caption_mask  # [batch_size, 5, length]
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

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for idx, device in enumerate(self.devices):
                with tf.device("/%s" % device):
                    if idx > 0:
                        tf.get_variable_scope().reuse_variables()

                    from_idx = self.batch_size_per_gpu*idx

                    video = tf.slice(self.video, [from_idx,0,0,0,0],
                                        [self.batch_size_per_gpu,-1,-1,-1,-1])
                    video_mask = tf.slice(self.video_mask, [from_idx,0],
                                            [self.batch_size_per_gpu,-1])
                    caption = tf.slice(self.caption, [from_idx,0],
                                            [self.batch_size_per_gpu,-1])
                    caption_mask = tf.slice(self.caption_mask, [from_idx,0],
                                                [self.batch_size_per_gpu,-1])
                    answer = tf.slice(self.answer, [from_idx,0],
                                                [self.batch_size_per_gpu,-1])

                    self.build_graph_single_gpu(video, video_mask, caption,
                                                caption_mask, answer, idx)

        self.mean_loss = tf.reduce_mean(tf.stack(self.mean_loss_list, axis=0))
        self.mean_grad = average_gradients(self.mean_grad_list) # use this to debug.
        self.alpha = tf.stack(self.alpha_list, axis=0)
        self.predictions = tf.stack(self.predictions_list, axis=0)
        self.correct_predictions = tf.stack(self.correct_predictions_list, axis=0)
        self.acc = tf.reduce_mean(tf.stack(self.acc_list, axis=0))


    def build_graph_single_gpu(self, video, video_mask, caption, caption_mask, answer, idx):

        with tf.variable_scope("word_emb"):
            with tf.device("/cpu:0"):
                embedded_captions = tf.nn.embedding_lookup(self.word_embed_t, caption)
                # [batch_size, length, word_dim]
                embedded_start_word = tf.nn.embedding_lookup(
                    self.word_embed_t, tf.ones([self.batch_size_per_gpu], dtype=tf.int32))


        with tf.variable_scope("caption_rnn") as scope:
            caption_cell = MultiRNNCell([self.get_rnn_cell()] * self.num_layers)
            cap_rnn_states = [caption_cell.zero_state(self.batch_size_per_gpu, tf.float32)]
            current_embedded_y = embedded_start_word

            for i in range(self.lstm_steps):
                if i > 0:
                    scope.reuse_variables()

                new_output, new_state = caption_cell(
                    current_embedded_y, cap_rnn_states[-1]
                )
                cap_rnn_states.append(new_state)
                current_embedded_y = embedded_captions[:, i, :]

        def spatio_att():
            with tf.variable_scope("merge_emb") as scope:
                rnn_final_state1 = tf.concat([cap_rnn_states[-1][0][1], cap_rnn_states[-1][1][1]], 1)

                fc_sent = linear(rnn_final_state1, 512, name="fc_sent")
                fc_sent_tiled = tf.tile(fc_sent, [1, self.lstm_steps*7*7])
                fc_sent_tiled = tf.reshape(fc_sent_tiled, [-1, 512])

                video_2d = tf.reshape(video, [self.batch_size_per_gpu*self.lstm_steps*7*7, self.channel_size])
                fc_vid = linear(video_2d, 512, name="fc_vid")
                pooled = tf.tanh(tf.add(fc_vid, fc_sent_tiled))

                pre_alpha = linear(pooled, 1, name="pre_alpha")
                pre_alpha = tf.reshape(pre_alpha, [-1, 7*7])
                alpha = tf.nn.softmax(pre_alpha)
                alpha = tf.reshape(alpha, [self.batch_size_per_gpu*self.lstm_steps, 7*7, 1])
                return alpha
        def const_att():
            return tf.constant(1./7*7, dtype=tf.float32, shape=[self.batch_size_per_gpu*self.lstm_steps, 7*7, 1])

        alpha = tf.cond(self.train_step < self.N_PRETRAIN, const_att, spatio_att)
        self.alpha_list.append(tf.reshape(alpha, [self.batch_size_per_gpu, self.lstm_steps, 7*7]))

        with tf.variable_scope("att_image_emb"):
            batch_pre_att = tf.reshape(video, [self.batch_size_per_gpu*self.lstm_steps,
                                                7*7, self.channel_size])
            embedded_feat = tf.reduce_sum(batch_pre_att * alpha, 1)
            embedded_feat = tf.reshape(embedded_feat, [self.batch_size_per_gpu, self.lstm_steps, self.channel_size])

            #  [batch_size, length, channel_size]
            embedded_feat_drop = tf.nn.dropout(
                embedded_feat, self.dropout_keep_prob_t)

        with tf.variable_scope("video_rnn") as scope:
            video_cell = MultiRNNCell([self.get_rnn_cell()] * self.num_layers)
            vid_rnn_states = [video_cell.zero_state(self.batch_size_per_gpu, tf.float32)]

            for i in range(self.lstm_steps):
                if i > 0:
                    scope.reuse_variables()
                new_output, new_state = video_cell(
                    embedded_feat_drop[:, i, :], vid_rnn_states[-1]
                )
                vid_rnn_states.append(
                    (
                        (new_state[0][0]*tf.expand_dims(video_mask[:, i], 1),
                        new_state[0][1]*tf.expand_dims(video_mask[:, i], 1)),
                        (new_state[1][0]*tf.expand_dims(video_mask[:, i], 1),
                        new_state[1][1]*tf.expand_dims(video_mask[:, i], 1))
                    )
                )

            vid_states = [
                tf.concat([vid_rnn_state[0][0], vid_rnn_state[1][0]], 1)
                for vid_rnn_state in vid_rnn_states[1:]
            ]

        with tf.variable_scope("caption_rnn") as scope:
            scope.reuse_variables()
            caption_cell = MultiRNNCell([self.get_rnn_cell()] * self.num_layers)
            cap_rnn_states = [vid_rnn_states[-1]]
            current_embedded_y = embedded_start_word

            for i in range(self.lstm_steps):
                if i > 0:
                    scope.reuse_variables()
                new_output, new_state = caption_cell(current_embedded_y, cap_rnn_states[-1])
                cap_rnn_states.append(new_state)
                current_embedded_y = embedded_captions[:, i, :]

        with tf.variable_scope("merge") as scope:
            rnn_final_state = tf.concat([cap_rnn_states[-1][0][1], cap_rnn_states[-1][1][1]], 1)
            vid_att, alpha = self.attention(rnn_final_state, vid_states)
            final_embed = tf.add(tf.nn.tanh(linear(vid_att, 2*self.hidden_dim)),
                                 rnn_final_state)

        with tf.variable_scope("loss") as scope:
            rnnW = tf.get_variable(
                "W",
                [2*self.hidden_dim, self.answer_size],
                initializer=tf.random_normal_initializer(stddev=0.1))
            rnnb = tf.get_variable(
                "b",
                [self.answer_size],
                initializer=tf.constant_initializer(0.0))
            embed_state = tf.nn.xw_plus_b(final_embed,rnnW,rnnb)

            labels = answer
            indices = tf.expand_dims(tf.range(0, self.batch_size_per_gpu, 1), 1)
            labels_with_index = tf.concat([indices, labels], 1)

            onehot_labels = tf.sparse_to_dense(labels_with_index,
                                                tf.stack([self.batch_size_per_gpu, self.answer_size]),
                                                sparse_values=1.0,
                                                default_value=0)
            cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=embed_state, labels=onehot_labels)

            mean_loss = tf.reduce_mean(cross_entropy_loss, name="t_loss")
            self.mean_loss_list.append(mean_loss)

        with tf.variable_scope("gradient") as scope:
            gs, vs = zip(*self.optimizer.compute_gradients(mean_loss))
            clipped_gs, _ = tf.clip_by_global_norm(gs, clip_norm=5)
            self.mean_grad_list.append(zip(clipped_gs, vs))


        with tf.variable_scope("accuracy"):
            # prediction tensor on test phase
            predictions = tf.argmax(
                tf.reshape(embed_state, [self.batch_size_per_gpu, self.answer_size]),
                dimension=1, name='argmax_predictions'
            )
            predictions.get_shape().assert_is_compatible_with([self.batch_size_per_gpu])

            correct_predictions = tf.cast(tf.equal(
                tf.reshape(predictions, [self.batch_size_per_gpu, 1]),
                tf.cast(answer,tf.int64)), tf.int32)
            acc = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy_%d"%idx)

            self.predictions_list.append(predictions)
            self.correct_predictions_list.append(correct_predictions)
            self.acc_list.append(acc)

    def attention(self, prev_hidden, vid_states):
        packed = tf.stack(vid_states)
        packed = tf.transpose(packed, [1,0,2])
        vid_2d = tf.reshape(packed, [-1, self.hidden_dim*2])
        sent_2d = tf.tile(prev_hidden, [1, self.lstm_steps])
        sent_2d = tf.reshape(sent_2d, [-1, self.hidden_dim*2])
        preact = tf.add(linear(sent_2d, self.hidden_dim, name="preatt_sent"),
                        linear(vid_2d, self.hidden_dim, name="preadd_vid"))
        score = linear(tf.nn.tanh(preact), 1, name="preatt")
        score_2d = tf.reshape(score, [-1, self.lstm_steps])
        alpha = tf.nn.softmax(score_2d)
        alpha_3d = tf.reshape(alpha, [-1, self.lstm_steps, 1])
        return tf.reduce_sum(packed * alpha_3d, 1), alpha

class FrameQASpTpEvaluator(FrameQABaseEvaluator):
    pass
class FrameQASpTpTrainer(FrameQABaseTrainer):
    pass

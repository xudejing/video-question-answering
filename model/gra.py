"""Gradually Refined Attention Network."""
import numpy as np
import tensorflow as tf


class GRA(object):
    """Build graph for Gradually Refined Attention Network."""

    def __init__(self, config):
        """Init model."""
        self.word_dim = config['word_dim']
        self.vocab_num = config['vocab_num']
        self.pretrained_embedding = config['pretrained_embedding']
        self.appear_dim = config['appear_dim']
        self.frame_num = config['frame_num']
        self.motion_dim = config['motion_dim']
        self.clip_num = config['clip_num']
        self.common_dim = config['common_dim']
        self.answer_num = config['answer_num']

        self.motion = None
        self.appear = None
        self.question_encode = None
        self.answer_encode = None

        self.appear_weight = None
        self.motion_weight = None
        self.channel_weight = None

        self.logit = None
        self.prediction = None
        self.loss = None
        self.acc = None

        self.train = None

    def build_inference(self):
        """Build inference graph."""
        with tf.name_scope('input'):
            self.appear = tf.placeholder(
                tf.float32, [None, self.frame_num, self.appear_dim], 'appear')
            self.motion = tf.placeholder(
                tf.float32, [None, self.clip_num, self.motion_dim], 'motion')
            self.question_encode = tf.placeholder(
                tf.int64, [None, None], 'question_encode')

        with tf.variable_scope('embedding'):
            if self.pretrained_embedding:
                embedding_matrix = tf.get_variable(
                    'embedding_matrix', initializer=np.load(self.pretrained_embedding),
                    regularizer=tf.nn.l2_loss)
            else:
                embedding_matrix = tf.get_variable(
                    'embedding_matrix',
                    [self.vocab_num, self.word_dim], regularizer=tf.nn.l2_loss)
            question_embedding = tf.nn.embedding_lookup(
                embedding_matrix, self.question_encode, name='word_embedding')

        with tf.variable_scope('transform_video'):
            with tf.variable_scope('appear'):
                W = tf.get_variable(
                    'W', [self.appear_dim, self.common_dim],
                    regularizer=tf.nn.l2_loss)
                b = tf.get_variable('b', [self.common_dim])
                appear = tf.reshape(self.appear, [-1, self.appear_dim])
                appear = tf.nn.xw_plus_b(appear, W, b)
                appear = tf.reshape(
                    appear, [-1, self.frame_num, self.common_dim])
                appear = tf.nn.tanh(appear)
            with tf.variable_scope('motion'):
                W = tf.get_variable(
                    'W', [self.motion_dim, self.common_dim],
                    regularizer=tf.nn.l2_loss)
                b = tf.get_variable('b', [self.common_dim])
                motion = tf.reshape(self.motion, [-1, self.motion_dim])
                motion = tf.nn.xw_plus_b(motion, W, b)
                motion = tf.reshape(
                    motion, [-1, self.clip_num, self.common_dim])
                motion = tf.nn.tanh(motion)

        with tf.variable_scope('init'):
            shape = tf.shape(self.question_encode)
            batch_size = shape[0]
            question_length = shape[1]
            time = tf.constant(0, name='time')

            q_cell = tf.nn.rnn_cell.BasicLSTMCell(self.word_dim)
            a_cell = tf.nn.rnn_cell.BasicLSTMCell(self.common_dim)
            q_state = q_cell.zero_state(batch_size, tf.float32)
            a_state = a_cell.zero_state(batch_size, tf.float32)

            appear_weight = tf.zeros([batch_size, self.frame_num])
            motion_weight = tf.zeros([batch_size, self.clip_num])
            channel_weight = tf.zeros([batch_size, 2])
            fused = tf.zeros([batch_size, self.common_dim])

            word_embed_W = tf.get_variable(
                'word_embed_W', [self.word_dim, self.common_dim],
                regularizer=tf.nn.l2_loss)
            word_embed_b = tf.get_variable(
                'word_embed_b', [self.common_dim])
            channel_W = tf.get_variable(
                'channel_W', [self.word_dim, 2],
                regularizer=tf.nn.l2_loss)
            channel_b = tf.get_variable('channel_b', [2])

        # Process one word
        def _one_step(time, q_state, a_state, appear_weight, motion_weight, channel_weight, fused):
            """One time step of model."""
            word_embedding = question_embedding[:, time]
            with tf.variable_scope('lstm_q'):
                q_output, q_state = q_cell(word_embedding, q_state)
            # map to common dimension
            with tf.name_scope('transform_w'):
                word = tf.nn.xw_plus_b(
                    word_embedding, word_embed_W, word_embed_b)
                word = tf.nn.tanh(word)
            with tf.name_scope('transform_q'):
                question = tf.nn.xw_plus_b(
                    q_output, word_embed_W, word_embed_b)
                question = tf.nn.tanh(question)

            with tf.variable_scope('amu'):
                with tf.name_scope('attend_1'):
                    appear_weight_1, appear_att_1 = self.attend(
                        word, appear, 'appear')
                    motion_weight_1, motion_att_1 = self.attend(
                        word, motion, 'motion')

                with tf.name_scope('channel_fuse'):
                    # word attend on channel
                    channel_weight = tf.nn.softmax(
                        tf.nn.xw_plus_b(word_embedding, channel_W, channel_b))
                    cw_appear = tf.expand_dims(channel_weight[:, 0], 1)
                    cw_motion = tf.expand_dims(channel_weight[:, 1], 1)
                    current_video_att = cw_appear * appear_att_1 + cw_motion * motion_att_1

                with tf.name_scope('sum'):
                    previous_video_att = fused
                    a_input = current_video_att + previous_video_att + question

                with tf.variable_scope('lstm_a'):
                    a_output, a_state = a_cell(a_input, a_state)

                with tf.name_scope('attend_2'):
                    appear_weight_2, _ = self.attend(
                        a_output, appear, 'appear')
                    motion_weight_2, _ = self.attend(
                        a_output, motion, 'motion')

                with tf.name_scope('refine'):
                    appear_weight = (appear_weight_1 + appear_weight_2) / 2
                    motion_weight = (motion_weight_1 + motion_weight_2) / 2

                    appear_att = tf.reduce_sum(
                        tf.expand_dims(appear_weight, 2) * appear, 1)
                    motion_att = tf.reduce_sum(
                        tf.expand_dims(motion_weight, 2) * motion, 1)

                    # question attend on channel
                    channel_weight = tf.nn.softmax(
                        tf.nn.xw_plus_b(q_output, channel_W, channel_b))
                    cw_appear = tf.expand_dims(channel_weight[:, 0], 1)
                    cw_motion = tf.expand_dims(channel_weight[:, 1], 1)
                    fused = cw_appear * appear_att + cw_motion * motion_att

            return time + 1, q_state, a_state, appear_weight, motion_weight, channel_weight, fused

        # main loop
        time, q_state, a_state, appear_weight, motion_weight, channel_weight, fused = tf.while_loop(
            cond=lambda time, *_: time < question_length,
            body=_one_step,
            loop_vars=[time, q_state, a_state, appear_weight, motion_weight, channel_weight, fused])

        self.appear_weight = appear_weight
        self.motion_weight = motion_weight
        self.channel_weight = channel_weight

        # Answer Generation.
        with tf.variable_scope('fuse'):
            with tf.name_scope('q_info'):
                q_info = tf.nn.tanh(tf.nn.xw_plus_b(
                    q_state.c, word_embed_W, word_embed_b))
            with tf.name_scope('a_info'):
                a_info = tf.nn.tanh(a_state.c)
            with tf.name_scope('video_info'):
                video_info = tf.nn.tanh(fused)
            fuse = q_info * a_info * video_info

        with tf.variable_scope('output'):
            W = tf.get_variable(
                'W', [self.common_dim, self.answer_num],
                regularizer=tf.nn.l2_loss)
            b = tf.get_variable('b', [self.answer_num])
            self.logit = tf.nn.softmax(
                tf.nn.xw_plus_b(fuse, W, b), name='logit')
            self.prediction = tf.argmax(self.logit, axis=1, name='prediction')

    def build_loss(self, reg_coeff, shu_coeff):
        """Compute loss and acc."""
        with tf.name_scope('answer'):
            self.answer_encode = tf.placeholder(
                tf.int64, [None], 'answer_encode')
            answer_one_hot = tf.one_hot(
                self.answer_encode, self.answer_num)
        with tf.name_scope('loss'):
            log_loss = tf.losses.log_loss(
                answer_one_hot, self.logit, scope='log_loss')
            reg_loss = tf.add_n(
                tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
            # fix channel selection
            shu_loss = tf.reduce_sum(
                tf.abs(self.channel_weight[:, 0] - self.channel_weight[:, 1]))
            self.loss = log_loss + reg_coeff * reg_loss + shu_coeff * shu_loss

        with tf.name_scope("acc"):
            correct = tf.equal(self.prediction, self.answer_encode)
            self.acc = tf.reduce_mean(tf.cast(correct, "float"))

    def build_train(self, learning_rate):
        """Add train operation."""
        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train = optimizer.minimize(self.loss)

    def attend(self, target, sources, name=None):
        """Use target to attend on sources. `target` and `sources` should have equal dim.

        Args:
            target: [None, target_dim].
            sources: [None, source_num, source_dim].
        Returns:
            weight: [None, source_num].
            att: [None, source_dim].
        """
        with tf.name_scope(name, 'attend'):
            weight = tf.nn.softmax(tf.reduce_sum(
                tf.expand_dims(target, 1) * sources, 2))
            att = tf.reduce_sum(
                tf.expand_dims(weight, 2) * sources, 1)
            return weight, att

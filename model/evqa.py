"""Extended VQA model."""
import numpy as np
import tensorflow as tf


class EVQA(object):
    """Implementation of E-VQA model."""

    def __init__(self, config):
        """Init model."""
        self.word_dim = config['word_dim']
        self.vocab_num = config['vocab_num']
        self.pretrained_embedding = config['pretrained_embedding']
        self.video_feature_dim = config['video_feature_dim']
        self.video_feature_num = config['video_feature_num']
        self.common_dim = config['common_dim']
        self.answer_num = config['answer_num']

        self.video_feature = None
        self.question_encode = None
        self.answer_encode = None

        self.logit = None
        self.prediction = None
        self.loss = None
        self.acc = None

        self.train = None

    def build_inference(self):
        """Build inference graph."""
        with tf.name_scope('input'):
            self.video_feature = tf.placeholder(
                tf.float32, [None, self.video_feature_num, self.video_feature_dim], 'video_feature')
            self.question_encode = tf.placeholder(
                tf.int64, [None, None], 'question_encode')

        with tf.variable_scope('process_video'):
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.video_feature_dim)
            _, video_state = tf.nn.dynamic_rnn(
                cell, self.video_feature, dtype=tf.float32, scope='lstm')

        with tf.variable_scope('process_question'):
            if self.pretrained_embedding:
                embedding_matrix = tf.get_variable(
                    'embedding_matrix',
                    initializer=np.load(self.pretrained_embedding),
                    regularizer=tf.nn.l2_loss)
            else:
                embedding_matrix = tf.get_variable(
                    'embedding_matrix', [self.vocab_num, self.word_dim],
                    regularizer=tf.nn.l2_loss)
            question_embedding = tf.nn.embedding_lookup(
                embedding_matrix, self.question_encode, name='question_embedding')
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.word_dim)
            _, question_state = tf.nn.dynamic_rnn(
                cell, question_embedding, dtype=tf.float32, scope='lstm')

        with tf.variable_scope('fuse'):
            with tf.variable_scope('video'):
                W = tf.get_variable(
                    'W', [self.video_feature_dim, self.common_dim],
                    regularizer=tf.nn.l2_loss)
                b = tf.get_variable('b', [self.common_dim])
                video_fuse = tf.nn.tanh(tf.nn.xw_plus_b(video_state.c, W, b))
            with tf.variable_scope('question'):
                W = tf.get_variable(
                    'W', [self.word_dim, self.common_dim],
                    regularizer=tf.nn.l2_loss)
                b = tf.get_variable('b', [self.common_dim])
                question_fuse = tf.nn.tanh(
                    tf.nn.xw_plus_b(question_state.c, W, b))
            fuse = video_fuse * question_fuse

        with tf.variable_scope('output'):
            W = tf.get_variable(
                'W', [self.common_dim, self.answer_num],
                regularizer=tf.nn.l2_loss)
            b = tf.get_variable('b', [self.answer_num])
            self.logit = tf.nn.softmax(
                tf.nn.xw_plus_b(fuse, W, b), name='logit')
            self.prediction = tf.argmax(self.logit, axis=1, name='prediction')

    def build_loss(self, reg_coeff):
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
            self.loss = log_loss + reg_coeff * reg_loss
        with tf.name_scope("acc"):
            correct = tf.equal(self.prediction, self.answer_encode)
            self.acc = tf.reduce_mean(tf.cast(correct, "float"))

    def build_train(self, learning_rate):
        """Add train operation."""
        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train = optimizer.minimize(self.loss)

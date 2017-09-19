"""Extended Memory Network."""
import numpy as np
import tensorflow as tf


class EMN(object):
    """Implementation of E-MN model."""

    def __init__(self, config):
        """Init model."""
        self.word_dim = config['word_dim']
        self.vocab_num = config['vocab_num']
        self.pretrained_embedding = config['pretrained_embedding']
        self.video_feature_dim = config['video_feature_dim']
        self.video_feature_num = config['video_feature_num']
        self.memory_dim = config['memory_dim']
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
                tf.int32, [None, None], 'question_encode')

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

        with tf.variable_scope('process_video'):
            cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.video_feature_dim)
            cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.video_feature_dim)
            batch_size = tf.shape(self.video_feature)[0]
            seq_len = tf.fill(
                [1, batch_size], tf.constant(self.video_feature_num, dtype=tf.int64))[0]
            bi_video, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, self.video_feature, seq_len, dtype=tf.float32, scope='bi_lstm')
            bi_video = (bi_video[0] + bi_video[1]) / 2

        with tf.variable_scope('memory'):
            A1 = tf.get_variable(
                'A1', [self.video_feature_dim, self.memory_dim],
                regularizer=tf.nn.l2_loss)
            C1A2 = tf.get_variable(
                'C1A2', [self.video_feature_dim, self.memory_dim],
                regularizer=tf.nn.l2_loss)
            C2A3 = tf.get_variable(
                'C2A3', [self.video_feature_dim, self.memory_dim],
                regularizer=tf.nn.l2_loss)
            C3 = tf.get_variable(
                'C3', [self.video_feature_dim, self.memory_dim],
                regularizer=tf.nn.l2_loss)
            B = tf.get_variable(
                'B', [self.word_dim, self.memory_dim],
                regularizer=tf.nn.l2_loss)

            u1 = tf.matmul(question_state.c, B, name='u1')

            with tf.name_scope('hop_1'):
                with tf.name_scope('in'):
                    mem_in = tf.reshape(bi_video, [-1, self.video_feature_dim])
                    mem_in = tf.matmul(mem_in, A1)
                    mem_in = tf.reshape(
                        mem_in, [-1, self.video_feature_num, self.memory_dim])
                    mem_in = tf.nn.tanh(mem_in)
                with tf.name_scope('out'):
                    mem_out = tf.reshape(
                        bi_video, [-1, self.video_feature_dim])
                    mem_out = tf.matmul(mem_out, C1A2)
                    mem_out = tf.reshape(
                        mem_out, [-1, self.video_feature_num, self.memory_dim])
                    mem_out = tf.nn.tanh(mem_out)
                weight, _ = self.attend(u1, mem_in)

            o1 = tf.reduce_sum(
                tf.expand_dims(weight, 2) * mem_out, 1, name='o1')
            u2 = tf.add(u1, o1, name='u2')

            with tf.name_scope('hop_2'):
                with tf.name_scope('in'):
                    mem_in = tf.reshape(bi_video, [-1, self.video_feature_dim])
                    mem_in = tf.matmul(mem_in, C1A2)
                    mem_in = tf.reshape(
                        mem_in, [-1, self.video_feature_num, self.memory_dim])
                    mem_in = tf.tanh(mem_in)
                with tf.name_scope('out'):
                    mem_out = tf.reshape(
                        bi_video, [-1, self.video_feature_dim])
                    mem_out = tf.matmul(mem_out, C2A3)
                    mem_out = tf.reshape(
                        mem_out, [-1, self.video_feature_num, self.memory_dim])
                    mem_out = tf.nn.tanh(mem_out)
                weight, _ = self.attend(u2, mem_in)

            o2 = tf.reduce_sum(
                tf.expand_dims(weight, 2) * mem_out, 1, name='o2')
            u3 = tf.add(u2, o2, name='u3')

            with tf.name_scope('hop_3'):
                with tf.name_scope('in'):
                    mem_in = tf.reshape(bi_video, [-1, self.video_feature_dim])
                    mem_in = tf.matmul(mem_in, C2A3)
                    mem_in = tf.reshape(
                        mem_in, [-1, self.video_feature_num, self.memory_dim])
                    mem_in = tf.tanh(mem_in)
                with tf.name_scope('out'):
                    mem_out = tf.reshape(
                        bi_video, [-1, self.video_feature_dim])
                    mem_out = tf.matmul(mem_out, C3)
                    mem_out = tf.reshape(
                        mem_out, [-1, self.video_feature_num, self.memory_dim])
                    mem_out = tf.nn.tanh(mem_out)
                weight, _ = self.attend(u3, mem_in)

            o3 = tf.reduce_sum(
                tf.expand_dims(weight, 2) * mem_out, 1, name='o3')
            fuse = tf.add(u3, o3, name='fuse')

        with tf.variable_scope('output'):
            W = tf.get_variable(
                'W', [self.memory_dim, self.answer_num],
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

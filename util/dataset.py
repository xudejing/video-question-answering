"""Generate data batch."""
import os

import pandas as pd
import numpy as np
import tables


class MSVDQA(object):
    """Use bucketing and padding to generate data batch.

    All questions are divided into 4 buckets and each buckets has its length.
    """

    def __init__(self, train_batch_size, preprocess_dir):
        """Load video feature, question answer, vocabulary, answer set.

        Note:
            The `train_batch_size` only impacts train. val and test batch size is 1.
        """
        self.train_batch_size = train_batch_size

        self.video_feature = tables.open_file(
            os.path.join(preprocess_dir, 'video_feature_20.h5'))

        # contains encode, for fast generate batch by saving encode time
        self.train_qa = pd.read_json(
            os.path.join(preprocess_dir, 'train_qa_encode.json'))
        self.val_qa = pd.read_json(
            os.path.join(preprocess_dir, 'val_qa_encode.json'))
        self.test_qa = pd.read_json(
            os.path.join(preprocess_dir, 'test_qa_encode.json'))

        # init train batch setting
        self.train_qa['question_length'] = self.train_qa.apply(
            lambda row: len(row['question'].split()), axis=1)
        self.train_buckets = [
            self.train_qa[(self.train_qa['question_length'] >= 2)
                          & (self.train_qa['question_length'] <= 6)],
            self.train_qa[(self.train_qa['question_length'] >= 7)
                          & (self.train_qa['question_length'] <= 11)],
            self.train_qa[(self.train_qa['question_length'] >= 12)
                          & (self.train_qa['question_length'] <= 16)],
            self.train_qa[(self.train_qa['question_length'] >= 17)
                          & (self.train_qa['question_length'] <= 21)]
        ]
        self.current_bucket = 0
        self.train_batch_length = [6, 11, 16, 21]
        self.train_batch_idx = [0, 0, 0, 0]
        self.train_batch_total = [
            len(self.train_buckets[0]) // train_batch_size,
            len(self.train_buckets[1]) // train_batch_size,
            len(self.train_buckets[2]) // train_batch_size,
            len(self.train_buckets[3]) // train_batch_size,
        ]
        # upset the arise of questions of same video
        for i in range(4):
            self.train_buckets[i] = self.train_buckets[i].sample(frac=1)
        self.has_train_batch = True

        # init val example setting
        self.val_example_total = len(self.val_qa)
        self.val_example_idx = 0
        self.has_val_example = True

        # init test example setting
        self.test_example_total = len(self.test_qa)
        self.test_example_idx = 0
        self.has_test_example = True

    def reset_train(self):
        """Reset train batch setting."""
        # random
        for i in range(4):
            self.train_buckets[i] = self.train_buckets[i].sample(frac=1)
        self.current_bucket = 0
        self.train_batch_idx = [0, 0, 0, 0]
        self.has_train_batch = True

    def reset_val(self):
        """Reset val batch setting."""
        self.val_example_idx = 0
        self.has_val_example = True

    def reset_test(self):
        """Reset train batch setting."""
        self.test_example_idx = 0
        self.has_test_example = True

    def get_train_batch(self):
        """Get [train_batch_size] examples as one train batch. Both question and answer
        are converted to int. The data batch is selected from short buckets to long buckets."""
        vgg_batch = []
        c3d_batch = []
        question_batch = []
        answer_batch = []

        bucket = self.train_buckets[self.current_bucket]
        start = self.train_batch_idx[
            self.current_bucket] * self.train_batch_size
        end = start + self.train_batch_size

        question_encode = bucket.iloc[start:end]['question_encode'].values
        answer_batch = bucket.iloc[start:end]['answer_encode'].values
        video_ids = bucket.iloc[start:end]['video_id'].values
        batch_length = self.train_batch_length[self.current_bucket]

        for i in range(self.train_batch_size):
            qid = [int(x) for x in question_encode[i].split(',')]
            qid = np.pad(qid, (0, batch_length - len(qid)), 'constant')
            question_batch.append(qid)
            vgg_batch.append(self.video_feature.root.vgg[video_ids[i] - 1])
            c3d_batch.append(self.video_feature.root.c3d[video_ids[i] - 1])

        self.train_batch_idx[self.current_bucket] += 1
        # if current bucket is ran out, use next bucket.
        if self.train_batch_idx[self.current_bucket] == self.train_batch_total[self.current_bucket]:
            self.current_bucket += 1

        if self.current_bucket == len(self.train_batch_total):
            self.has_train_batch = False

        return vgg_batch, c3d_batch, question_batch, answer_batch

    def get_val_example(self):
        """Get one val example. Only question is converted to int."""
        question_encode = self.val_qa.iloc[
            self.val_example_idx]['question_encode']
        video_id = self.val_qa.iloc[self.val_example_idx]['video_id']
        answer = self.val_qa.iloc[self.val_example_idx]['answer']

        question = [int(x) for x in question_encode.split(',')]
        vgg = self.video_feature.root.vgg[video_id - 1]
        c3d = self.video_feature.root.c3d[video_id - 1]

        self.val_example_idx += 1
        if self.val_example_idx == self.val_example_total:
            self.has_val_example = False

        return vgg, c3d, question, answer

    def get_test_example(self):
        """Get one test example. Only question is converted to int."""
        example_id = self.test_qa.iloc[self.test_example_idx]['id']
        question_encode = self.test_qa.iloc[
            self.test_example_idx]['question_encode']
        video_id = self.test_qa.iloc[self.test_example_idx]['video_id']
        answer = self.test_qa.iloc[self.test_example_idx]['answer']

        question = [int(x) for x in question_encode.split(',')]
        vgg = self.video_feature.root.vgg[video_id - 1]
        c3d = self.video_feature.root.c3d[video_id - 1]

        self.test_example_idx += 1
        if self.test_example_idx == self.test_example_total:
            self.has_test_example = False

        return vgg, c3d, question, answer, example_id


class MSRVTTQA(object):
    """Use bucketing and padding to generate data batch.

    All questions are divided into 5 buckets and each buckets has its length.
    """

    def __init__(self, train_batch_size, preprocess_dir):
        """Load video feature, question answer, vocabulary, answer set.

        Note:
            The `train_batch_size` only impacts train. val and test batch size is 1.
        """
        self.train_batch_size = train_batch_size

        self.video_feature = tables.open_file(
            os.path.join(preprocess_dir, 'video_feature_20.h5'))

        # contains encode, for fast generate batch by saving encode time
        self.train_qa = pd.read_json(
            os.path.join(preprocess_dir, 'train_qa_encode.json'))
        self.val_qa = pd.read_json(
            os.path.join(preprocess_dir, 'val_qa_encode.json'))
        self.test_qa = pd.read_json(
            os.path.join(preprocess_dir, 'test_qa_encode.json'))

        # init train batch setting
        self.train_qa['question_length'] = self.train_qa.apply(
            lambda row: len(row['question'].split()), axis=1)
        self.train_buckets = [
            self.train_qa[(self.train_qa['question_length'] >= 2)
                          & (self.train_qa['question_length'] <= 6)],
            self.train_qa[(self.train_qa['question_length'] >= 7)
                          & (self.train_qa['question_length'] <= 11)],
            self.train_qa[(self.train_qa['question_length'] >= 12)
                          & (self.train_qa['question_length'] <= 16)],
            self.train_qa[(self.train_qa['question_length'] >= 17)
                          & (self.train_qa['question_length'] <= 21)],
            self.train_qa[(self.train_qa['question_length'] >= 22)
                          & (self.train_qa['question_length'] <= 26)]
        ]
        self.current_bucket = 0
        self.train_batch_length = [6, 11, 16, 21, 26]
        self.train_batch_idx = [0, 0, 0, 0, 0]
        self.train_batch_total = [
            len(self.train_buckets[0]) // train_batch_size,
            len(self.train_buckets[1]) // train_batch_size,
            len(self.train_buckets[2]) // train_batch_size,
            len(self.train_buckets[3]) // train_batch_size,
            len(self.train_buckets[4]) // train_batch_size,
        ]
        # upset the arise of questions of same video
        for i in range(5):
            self.train_buckets[i] = self.train_buckets[i].sample(frac=1)
        self.has_train_batch = True

        # init val example setting
        self.val_example_total = len(self.val_qa)
        self.val_example_idx = 0
        self.has_val_example = True

        # init test example setting
        self.test_example_total = len(self.test_qa)
        self.test_example_idx = 0
        self.has_test_example = True

    def reset_train(self):
        """Reset train batch setting."""
        # random
        for i in range(5):
            self.train_buckets[i] = self.train_buckets[i].sample(frac=1)
        self.current_bucket = 0
        self.train_batch_idx = [0, 0, 0, 0, 0]
        self.has_train_batch = True

    def reset_val(self):
        """Reset val batch setting."""
        self.val_example_idx = 0
        self.has_val_example = True

    def reset_test(self):
        """Reset train batch setting."""
        self.test_example_idx = 0
        self.has_test_example = True

    def get_train_batch(self):
        """Get [train_batch_size] examples as one train batch. Both question and answer
        are converted to int. The data batch is selected from short buckets to long buckets."""
        vgg_batch = []
        c3d_batch = []
        question_batch = []
        answer_batch = []

        bucket = self.train_buckets[self.current_bucket]
        start = self.train_batch_idx[
            self.current_bucket] * self.train_batch_size
        end = start + self.train_batch_size

        question_encode = bucket.iloc[start:end]['question_encode'].values
        answer_batch = bucket.iloc[start:end]['answer_encode'].values
        video_ids = bucket.iloc[start:end]['video_id'].values
        batch_length = self.train_batch_length[self.current_bucket]

        for i in range(self.train_batch_size):
            qid = [int(x) for x in question_encode[i].split(',')]
            qid = np.pad(qid, (0, batch_length - len(qid)), 'constant')
            question_batch.append(qid)
            vgg_batch.append(self.video_feature.root.vgg[video_ids[i]])
            c3d_batch.append(self.video_feature.root.c3d[video_ids[i]])

        self.train_batch_idx[self.current_bucket] += 1
        # if current bucket is ran out, use next bucket.
        if self.train_batch_idx[self.current_bucket] == self.train_batch_total[self.current_bucket]:
            self.current_bucket += 1

        if self.current_bucket == len(self.train_batch_total):
            self.has_train_batch = False

        return vgg_batch, c3d_batch, question_batch, answer_batch

    def get_val_example(self):
        """Get one val example. Only question is converted to int."""
        question_encode = self.val_qa.iloc[
            self.val_example_idx]['question_encode']
        video_id = self.val_qa.iloc[self.val_example_idx]['video_id']
        answer = self.val_qa.iloc[self.val_example_idx]['answer']

        question = [int(x) for x in question_encode.split(',')]
        vgg = self.video_feature.root.vgg[video_id]
        c3d = self.video_feature.root.c3d[video_id]

        self.val_example_idx += 1
        if self.val_example_idx == self.val_example_total:
            self.has_val_example = False

        return vgg, c3d, question, answer

    def get_test_example(self):
        """Get one test example. Only question is converted to int."""
        example_id = self.test_qa.iloc[self.test_example_idx]['id']
        question_encode = self.test_qa.iloc[
            self.test_example_idx]['question_encode']
        video_id = self.test_qa.iloc[self.test_example_idx]['video_id']
        answer = self.test_qa.iloc[self.test_example_idx]['answer']

        question = [int(x) for x in question_encode.split(',')]
        vgg = self.video_feature.root.vgg[video_id]
        c3d = self.video_feature.root.c3d[video_id]

        self.test_example_idx += 1
        if self.test_example_idx == self.test_example_total:
            self.has_test_example = False

        return vgg, c3d, question, answer, example_id

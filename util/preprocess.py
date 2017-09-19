"""Preprocess the data for model."""
import os
import inspect
import csv

import numpy as np
from PIL import Image
import skvideo.io
import scipy
import tensorflow as tf
import pandas as pd

from .vgg16 import Vgg16
from .c3d import c3d


class VideoVGGExtractor(object):
    """Select uniformly distributed frames and extract its VGG feature."""

    def __init__(self, frame_num, sess):
        """Load VGG model.

        Args:
            frame_num: number of frames per video.
            sess: tf.Session()
        """
        self.frame_num = frame_num
        self.inputs = tf.placeholder(tf.float32, [self.frame_num, 224, 224, 3])
        self.vgg16 = Vgg16()
        self.vgg16.build(self.inputs)
        self.sess = sess

    def _select_frames(self, path):
        """Select representative frames for video.

        Ignore some frames both at begin and end of video.

        Args:
            path: Path of video.
        Returns:
            frames: list of frames.
        """
        frames = list()
        # video_info = skvideo.io.ffprobe(path)
        video_data = skvideo.io.vread(path)
        total_frames = video_data.shape[0]
        # Ignore some frame at begin and end.
        for i in np.linspace(0, total_frames, self.frame_num + 2)[1:self.frame_num + 1]:
            frame_data = video_data[int(i)]
            img = Image.fromarray(frame_data)
            img = img.resize((224, 224), Image.BILINEAR)
            frame_data = np.array(img)
            frames.append(frame_data)
        return frames

    def extract(self, path):
        """Get VGG fc7 activations as representation for video.

        Args:
            path: Path of video.
        Returns:
            feature: [batch_size, 4096]
        """
        frames = self._select_frames(path)
        # We usually take features after the non-linearity, by convention.
        feature = self.sess.run(
            self.vgg16.relu7, feed_dict={self.inputs: frames})
        return feature


class VideoC3DExtractor(object):
    """Select uniformly distributed clips and extract its C3D feature."""

    def __init__(self, clip_num, sess):
        """Load C3D model."""
        self.clip_num = clip_num
        self.inputs = tf.placeholder(
            tf.float32, [self.clip_num, 16, 112, 112, 3])
        _, self.c3d_features = c3d(self.inputs, 1, clip_num)
        saver = tf.train.Saver()
        path = inspect.getfile(VideoC3DExtractor)
        path = os.path.abspath(os.path.join(path, os.pardir))
        saver.restore(sess, os.path.join(
            path, 'sports1m_finetuning_ucf101.model'))
        self.mean = np.load(os.path.join(path, 'crop_mean.npy'))
        self.sess = sess

    def _select_clips(self, path):
        """Select self.batch_size clips for video. Each clip has 16 frames.

        Args:
            path: Path of video.
        Returns:
            clips: list of clips.
        """
        clips = list()
        # video_info = skvideo.io.ffprobe(path)
        video_data = skvideo.io.vread(path)
        total_frames = video_data.shape[0]
        height = video_data[1]
        width = video_data.shape[2]
        for i in np.linspace(0, total_frames, self.clip_num + 2)[1:self.clip_num + 1]:
            # Select center frame first, then include surrounding frames
            clip_start = int(i) - 8
            clip_end = int(i) + 8
            if clip_start < 0:
                clip_end = clip_end - clip_start
                clip_start = 0
            if clip_end > total_frames:
                clip_start = clip_start - (clip_end - total_frames)
                clip_end = total_frames
            clip = video_data[clip_start:clip_end]
            new_clip = []
            for j in range(16):
                frame_data = clip[j]
                img = Image.fromarray(frame_data)
                img = img.resize((112, 112), Image.BILINEAR)
                frame_data = np.array(img) * 1.0
                frame_data -= self.mean[j]
                new_clip.append(frame_data)
            clips.append(new_clip)
        return clips

    def extract(self, path):
        """Get 4096-dim activation as feature for video.

        Args:
            path: Path of video.
        Returns:
            feature: [self.batch_size, 4096]
        """
        clips = self._select_clips(path)
        feature = self.sess.run(
            self.c3d_features, feed_dict={self.inputs: clips})
        return feature


def prune_embedding(vocab_path, glove_path, embedding_path):
    """Prune word embedding from pre-trained GloVe.

    For words not included in GloVe, set to average of found embeddings.

    Args:
        vocab_path: vocabulary path.
        glove_path: pre-trained GLoVe word embedding.
        embedding_path: .npy for vocabulary embedding.
    """
    # load GloVe embedding.
    glove = pd.read_csv(
        glove_path, sep=' ', quoting=csv.QUOTE_NONE, header=None)
    glove.set_index(0, inplace=True)
    # load vocabulary.
    vocab = pd.read_csv(vocab_path, header=None)[0]

    embedding = np.zeros([len(vocab), len(glove.columns)], np.float64)
    not_found = []
    for i in range(len(vocab)):
        word = vocab[i]
        if word in glove.index:
            embedding[i] = glove.loc[word]
        else:
            not_found.append(i)
    print('Not found:\n', vocab.iloc[not_found])

    embedding_avg = np.mean(embedding, 0)
    embedding[not_found] = embedding_avg

    np.save(embedding_path, embedding.astype(np.float32))

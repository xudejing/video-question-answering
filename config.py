"""Configuration for models."""
import os

import tensorflow as tf

CONFIG = {
    'esa': {
        'msvd_qa': {
            '0': {
                'model': {
                    'word_dim': 300,
                    'vocab_num': 4000,
                    'pretrained_embedding': 'data/msvd_qa/word_embedding.npy',
                    'video_feature_dim': 4096,
                    'video_feature_num': 20,
                    'answer_num': 1000,
                    'common_dim': 256
                },
                'train': {
                    'batch_size': 32,
                    'reg_coeff': 1e-5,
                    'learning_rate': 0.001
                }
            }
        },
        'msrvtt_qa': {
            '0': {
                'model': {
                    'word_dim': 300,
                    'vocab_num': 8000,
                    'pretrained_embedding': 'data/msrvtt_qa/word_embedding.npy',
                    'video_feature_dim': 4096,
                    'video_feature_num': 20,
                    'answer_num': 1000,
                    'common_dim': 256
                },
                'train': {
                    'batch_size': 64,
                    'reg_coeff': 1e-6,
                    'learning_rate': 0.001
                }
            }
        }
    },
    'emn': {
        'msvd_qa': {
            '0': {
                'model': {
                    'word_dim': 300,
                    'vocab_num': 4000,
                    'pretrained_embedding': 'data/msvd_qa/word_embedding.npy',
                    'video_feature_dim': 4096,
                    'video_feature_num': 20,
                    'answer_num': 1000,
                    'memory_dim': 256
                },
                'train': {
                    'batch_size': 32,
                    'reg_coeff': 1e-5,
                    'learning_rate': 0.001
                }
            }
        },
        'msrvtt_qa': {
            '0': {
                'model': {
                    'word_dim': 300,
                    'vocab_num': 8000,
                    'pretrained_embedding': 'data/msrvtt_qa/word_embedding.npy',
                    'video_feature_dim': 4096,
                    'video_feature_num': 20,
                    'answer_num': 1000,
                    'memory_dim': 256
                },
                'train': {
                    'batch_size': 64,
                    'reg_coeff': 1e-6,
                    'learning_rate': 0.001
                }
            }
        }
    },
    'evqa': {
        'msvd_qa': {
            '0': {
                'model': {
                    'word_dim': 300,
                    'vocab_num': 4000,
                    'pretrained_embedding': 'data/msvd_qa/word_embedding.npy',
                    'video_feature_dim': 4096,
                    'video_feature_num': 20,
                    'answer_num': 1000,
                    'common_dim': 256
                },
                'train': {
                    'batch_size': 32,
                    'reg_coeff': 1e-5,
                    'learning_rate': 0.001
                }
            }
        },
        'msrvtt_qa': {
            '0': {
                'model': {
                    'word_dim': 300,
                    'vocab_num': 8000,
                    'pretrained_embedding': 'data/msrvtt_qa/word_embedding.npy',
                    'video_feature_dim': 4096,
                    'video_feature_num': 20,
                    'answer_num': 1000,
                    'common_dim': 256
                },
                'train': {
                    'batch_size': 64,
                    'reg_coeff': 1e-6,
                    'learning_rate': 0.001
                }
            }
        }
    },
    'gra': {
        'msvd_qa': {
            '0': {
                'model': {
                    'word_dim': 300,
                    'vocab_num': 4000,
                    'pretrained_embedding': 'data/msvd_qa/word_embedding.npy',
                    'appear_dim': 4096,
                    'frame_num': 20,
                    'motion_dim': 4096,
                    'clip_num': 20,
                    'answer_num': 1000,
                    'common_dim': 256
                },
                'train': {
                    'batch_size': 32,
                    'reg_coeff': 1e-6,
                    'shu_coeff': 1e-5,
                    'learning_rate': 0.001
                }
            }
        },
        'msrvtt_qa': {
            '0': {
                'model': {
                    'word_dim': 300,
                    'vocab_num': 8000,
                    'pretrained_embedding': 'data/msrvtt_qa/word_embedding.npy',
                    'appear_dim': 4096,
                    'frame_num': 20,
                    'motion_dim': 4096,
                    'clip_num': 20,
                    'answer_num': 1000,
                    'common_dim': 256
                },
                'train': {
                    'batch_size': 64,
                    'reg_coeff': 1e-7,
                    'shu_coeff': 1e-7,
                    'learning_rate': 0.001
                }
            }
        }
    }

}


def get(model, dataset, config_id, gpu_list):
    """Generate configuration."""
    config = {}
    if dataset == 'msvd_qa':
        config['preprocess_dir'] = 'data/msvd_qa'
    elif dataset == 'msrvtt_qa':
        config['preprocess_dir'] = 'data/msrvtt_qa'

    config['model'] = CONFIG[model][dataset][config_id]['model']
    config['train'] = CONFIG[model][dataset][config_id]['train']

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.visible_device_list = gpu_list
    config['session'] = sess_config

    return config

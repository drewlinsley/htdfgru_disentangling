import os
import numpy as np
import tensorflow as tf
from config import Config
from ops import tf_fun
from utils import image_processing
from glob import glob
from tqdm import tqdm
from skimage import io
try:
    import pandas as pd
except Exception:
    print('Failed to import pandas.')


class data_processing(object):
    def __init__(self):
        self.name = 'multicue_edges'
        self.output_name = 'multicue_edges'
        self.image_dir = '/media/data_cifs/pytorch_projects/datasets/Multicue_crops/data/images'
        self.im_extension = '.jpg'
        self.label_regex = r'(?<=length)\d+'
        self.config = Config()
        self.im_size = [500, 500, 3]  # 600, 600
        self.model_input_image_size = [320, 400, 3]  # [224, 224, 3]
        self.val_model_input_image_size = [320, 400, 3]
        self.output_size = [320, 400, 1]  # [321, 481, 1]
        self.label_size = self.output_size
        self.default_loss_function = 'cce'
        self.score_metric = 'accuracy'
        self.store_z = False
        self.normalize_im = False
        self.all_flips = True
        self.balance = True
        self.shuffle = True
        self.calculate_moments = False
        self.input_normalization = 'none'  # 'zscore'
        self.preprocess = []
        self.folds = {
            'train': 'train',
            'val': 'val',
            'test': 'test'
        }
        self.cv_split = 0.1
        self.cv_balance = True
        self.targets = {
            'image': tf_fun.bytes_feature,
            'label': tf_fun.bytes_feature
        }
        self.tf_dict = {
            'image': tf_fun.fixed_len_feature(dtype='string'),
            'label': tf_fun.fixed_len_feature(dtype='string')
        }
        self.tf_reader = {
            'image': {
                'dtype': tf.float32,
                'reshape': self.im_size
            },
            'label': {
                'dtype': tf.float32,
                'reshape': self.output_size
            }
        }

    def get_data(self):
        """Get the names of files."""

        all_ims = glob(
            os.path.join(
                self.image_dir,
                'train',
                '*%s' % self.im_extension))
        all_ims = np.array(all_ims)
        all_labels = [x.replace('images', 'groundTruth') for x in all_ims]
        all_labels = [x.replace('.jpg', '.edges.npy') for x in all_labels]
        all_labels = np.array(all_labels)
        assert len(all_ims) == len(all_labels)

        # Get test images
        test_ims = glob(
            os.path.join(
                self.image_dir,
                'test_nocrop',
                '*%s' % self.im_extension))
        test_labels = [x.replace('images', 'groundTruth') for x in test_ims]
        test_labels = [x.replace('.jpg', '.edges.npy') for x in test_labels]
        test_labels = np.array(test_labels)

        # Split into CV sets
        val_idx = np.random.permutation(
            len(all_labels)) < (len(all_labels) * self.cv_split)
        train_ims = all_ims[~val_idx]
        train_labels = all_labels[~val_idx]
        val_ims = all_ims[val_idx]
        val_labels = all_labels[val_idx]

        # Build CV dict
        cv_files, cv_labels = {}, {}
        cv_files[self.folds['train']] = train_ims
        cv_files[self.folds['val']] = val_ims
        cv_files[self.folds['test']] = val_ims
        cv_labels[self.folds['train']] = train_labels
        cv_labels[self.folds['val']] = val_labels
        cv_labels[self.folds['test']] = val_labels
        return cv_files, cv_labels

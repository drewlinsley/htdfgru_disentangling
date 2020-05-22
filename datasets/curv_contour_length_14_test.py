import os
import numpy as np
import tensorflow as tf
from config import Config
from ops import tf_fun


class data_processing(object):
    def __init__(self):
        self.name = 'curv_contour_length_14_test'
        self.output_name = 'curv_contour_length_14_test'
        self.data_name = 'curv_contour_length_14'
        self.contour_dir = '/media/data_cifs/pathfinder_small/'
        self.im_extension = '.png'
        self.label_regex = r'(?<=length)\d+'
        self.config = Config()
        self.im_size = [300, 300]  # 600, 600
        self.model_input_image_size = [160, 160, 1]  # [107, 160, 3]
        self.max_ims = 5000
        self.output_size = [1]
        self.label_size = self.output_size
        self.default_loss_function = 'cce'
        self.score_metric = 'accuracy'
        self.store_z = False
        self.normalize_im = False
        self.all_flips = True
        self.balance = True
        self.shuffle = True
        self.input_normalization = 'none'  # 'zscore'
        self.preprocess = ['']  # ['resize_nn']
        self.meta = os.path.join('metadata', 'combined.npy')
        self.negative = 'curv_contour_length_14_neg'
        self.folds = {
            'test': 'test',
        }
        self.cv_split = 0.9
        self.cv_balance = True
        self.targets = {
            'image': tf_fun.bytes_feature,
            'label': tf_fun.int64_feature
        }
        self.tf_dict = {
            'image': tf_fun.fixed_len_feature(dtype='string'),
            'label': tf_fun.fixed_len_feature(dtype='int64')
        }
        self.tf_reader = {
            'image': {
                'dtype': tf.float32,
                'reshape': self.im_size
            },
            'label': {
                'dtype': tf.int64,
                'reshape': self.output_size
            }
        }

    def list_files(self, meta, directory):
        """List files from metadata."""
        files = []
        for f in meta:
            files += [
                os.path.join(
                    self.contour_dir,
                    directory,
                    f[0],
                    f[1])]
        return np.asarray(files)

    def get_data(self):
        """Get the names of files."""
        positive_meta = np.load(
            os.path.join(
                self.contour_dir,
                self.data_name,
                self.meta))
        positive_ims = self.list_files(positive_meta, self.data_name)
        all_ims = positive_ims[
            np.random.permutation(len(positive_ims))]
        all_labels = np.ones(len(positive_ims)).astype(int)

        # Build CV dict
        cv_files, cv_labels = {}, {}
        cv_files[self.folds['test']] = all_ims
        cv_labels[self.folds['test']] = all_labels
        return cv_files, cv_labels

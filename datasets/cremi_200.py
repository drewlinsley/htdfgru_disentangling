import os
import numpy as np
import tensorflow as tf
from config import Config
from ops import tf_fun
from skimage import io


class data_processing(object):
    def __init__(self):
        self.name = 'full_cremi_200'
        self.output_name = 'full_cremi_200'
        self.contour_dir = '/media/data_cifs/connectomics/datasets/cremi_a_0.npz'
        self.config = Config()
        self.im_size = [256, 256]  # 600, 600
        self.model_input_image_size = [256, 256, 1]  # [107, 160, 3]
        self.nhot_size = [26]
        self.max_ims = 222
        self.output_size = {'output': 2, 'aux': self.nhot_size[0]}
        self.label_size = self.im_size
        self.default_loss_function = 'cce'
        # self.aux_loss = {'nhot': ['bce', 1.]}  # Loss type and scale
        self.score_metric = 'prop_positives'
        self.store_z = False
        self.normalize_im = False
        self.all_flips = True
        self.shuffle = True
        self.input_normalization = 'none'  # 'zscore'
        self.meta = os.path.join('metadata', 'combined.npy')
        self.folds = {
            'train': 'train',
            'test': 'test'
        }
        self.cv_split = 0.1
        self.train_size = 200
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
                'reshape': self.label_size
            }
        }

    def list_files(self, meta, directory, cat=0):
        """List files from metadata."""
        files, labs = [], []
        for f in meta:
            files += [
                os.path.join(
                    self.contour_dir,
                    directory,
                    f[0],
                    f[1])]
            labs += [
                os.path.join(
                    self.contour_dir,
                    directory,
                    f[2],
                    f[3])]
        return np.asarray(files), np.asarray(labs)

    def get_data(self):
        """Get the names of files."""
        d = np.load(self.contour_dir)
        volume = d['volume'][:, :1024, :1024]
        label = d['label'][:, :1024, :1024]
        window_size = 256
        strides = np.arange(0, volume.shape[1], window_size)
        window_volume, window_label = [], []
        for vol, lab in zip(volume, label):
            for x_stride_start in strides:
               for y_stride_start in strides:
                    x_stride_end = x_stride_start + window_size
                    y_stride_end = y_stride_start + window_size
                    window_volume += [vol[y_stride_start:y_stride_end, x_stride_start:x_stride_end]]
                    window_label += [lab[y_stride_start:y_stride_end, x_stride_start:x_stride_end]]
        window_volume = np.array(window_volume)
        window_label = np.array(window_label)

        # Balance CV folds
        val_cutoff = -200  # np.round(num_ims * self.cv_split).astype(int)
        train_ims = window_volume[:self.train_size]
        train_labels = window_label[:self.train_size]
        val_ims = window_volume[val_cutoff:]
        val_labels = window_label[val_cutoff:]

        # # Preload test data
        # val_ims = np.array([io.imread(x) for x in val_ims])
        # val_labels = np.array([io.imread(x) for x in val_labels])

        # Build CV dict
        cv_files, cv_labels = {}, {}
        cv_files[self.folds['train']] = val_ims
        cv_files[self.folds['test']] = val_ims
        cv_labels[self.folds['train']] = val_labels
        cv_labels[self.folds['test']] = val_labels
        return cv_files, cv_labels


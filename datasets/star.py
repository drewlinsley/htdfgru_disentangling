import os
import numpy as np
import tensorflow as tf
from config import Config
from ops import tf_fun


class data_processing(object):
    def __init__(self):
        self.name = 'star'
        self.output_name = 'star'
        self.contour_dir = ['/media/data_cifs/connectomics/datasets/isbi_2012_0.npz', '/media/data_cifs/connectomics/datasets/trim_fib253d_0.npz']
        self.config = Config()
        self.affinity = False
        self.im_size = [512, 512]  # 600, 600
        self.model_input_image_size = [384, 384, 1]  # [107, 160, 3]
        self.nhot_size = [26]
        self.max_ims = 222
        self.output_size = {'output': 1, 'aux': self.nhot_size[0]}
        self.label_size = self.im_size + [1]
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
            'val': 'val'
        }
        self.train_split = 0.8
        self.train_size = 1280
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

    def gather_windows(
            self,
            volume,
            label,
            window_size=512,
            stride_size=0,
            affinity=True):
        """Im2Col the data."""
        if window_size == volume.shape[1]:
            strides = [0]
        else:
            strides = np.arange(0, volume.shape[1] - stride_size, stride_size)
        window_volume, window_label = [], []
        for vol, lab in zip(volume, label):
            for x_stride_start in strides:
                for y_stride_start in strides:
                    x_stride_end = x_stride_start + window_size
                    y_stride_end = y_stride_start + window_size
                    window_volume += [
                        vol[
                            y_stride_start:y_stride_end,
                            x_stride_start:x_stride_end]]
                    window_label += [
                        lab[
                            y_stride_start:y_stride_end,
                            x_stride_start:x_stride_end]]
        window_volume = np.array(window_volume)
        window_label = np.array(window_label)
        if self.affinity:
            window_label = tf_fun.derive_affinities(
                window_label, long_range=True, use_3d=False)
        return window_volume, window_label

    def get_data(self):
        """Get the names of files."""
        train_ims, train_labels, val_ims, val_labels = [], [], [], []
        for p in self.contour_dir:
            d = np.load(p)
            volume = d['volume']
            label = d['label']
            train_split = int(len(volume) * self.train_split)
            train_im, train_label = self.gather_windows(
                volume[:train_split],
                label[:train_split])
            val_im, val_label = self.gather_windows(
                volume[train_split:],
                label[train_split:])
            train_ims += [train_im]
            train_labels += [train_label]
            val_ims += [val_im]
            val_labels += [val_label]

        # Build CV dict
        train_ims = np.concatenate(train_ims, 0)
        train_labels = np.concatenate(train_labels, 0)
        rand_idx = np.random.permutation(len(train_ims))
        train_ims = train_ims[rand_idx]
        train_labels = train_labels[rand_idx]
        cv_files, cv_labels = {}, {}
        cv_files[self.folds['train']] = train_ims
        cv_files[self.folds['val']] = np.concatenate(val_ims, 0)
        cv_labels[self.folds['train']] = train_labels
        cv_labels[self.folds['val']] = np.concatenate(val_labels, 0)
        return cv_files, cv_labels


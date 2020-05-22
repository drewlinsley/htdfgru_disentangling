import os
import numpy as np
import tensorflow as tf
from config import Config
from ops import tf_fun


class data_processing(object):
    def __init__(self):
        self.name = 'berson_1_affinity'
        self.output_name = 'berson_1_affinity'
        self.contour_dir = '/media/data_cifs/connectomics/datasets/berson_0.npz'
        self.config = Config()
        self.affinity = True
        self.im_size = [384, 384]  # 600, 600
        self.model_input_image_size = [200, 200, 1]  # [107, 160, 3]
        self.nhot_size = [26]
        self.max_ims = 222
        self.output_size = {'output': 4, 'aux': self.nhot_size[0]}
        self.label_size = self.im_size + [4]
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
        self.train_split = int(384 * .01)
        self.val_split = 307
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
            window_size=384,
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
                window_label,
                long_range=self.label_size[-1] == 8, use_3d=False)
        return window_volume, window_label

    def get_data(self):
        """Get the names of files."""
        d = np.load(self.contour_dir)
        volume = d['volume']
        label = d['label']
        train_ims, train_labels = self.gather_windows(
            volume[:self.train_split],
            label[:self.train_split])
        val_ims, val_labels = self.gather_windows(
            volume[self.val_split:],
            label[self.val_split:])

        # Build CV dict
        cv_files, cv_labels = {}, {}
        cv_files[self.folds['train']] = train_ims
        cv_files[self.folds['val']] = val_ims
        cv_labels[self.folds['train']] = train_labels
        cv_labels[self.folds['val']] = val_labels
        return cv_files, cv_labels

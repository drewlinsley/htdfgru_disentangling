import os
import numpy as np
import tensorflow as tf
from config import Config
from ops import tf_fun


class data_processing(object):
    def __init__(self):
        self.output_name = 'cardena'
        self.config = Config()
        self.im_size = [140, 140]  # 600, 600
        self.model_input_image_size = [140, 140, 1]  # [107, 160, 3]
        self.nhot_size = [26]
        self.max_ims = 222
        self.output_size = {'output': 166 * 1, 'aux': self.nhot_size[0]}
        self.label_size = [166 * 1]
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
        self.train_split = 0.9
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
        images = np.load('/media/data_cifs/image_datasets/cardena/images.npy')
        responses = np.load('/media/data_cifs/image_datasets/cardena/responses.npy')
        session_id = np.load('/media/data_cifs/image_datasets/cardena/session_id.npy')
        subject_id = np.load('/media/data_cifs/image_datasets/cardena/subject_id.npy')
        # labels = np.concatenate(
        #     (
                # responses,
                # session_id[None].repeat(len(responses), 0),
                # subject_id[None].repeat(len(responses), 0)), axis=-1)
        labels = responses
        labels = labels.astype(np.float32)
        rand_idx = np.random.permutation(len(images))
        train_idx = rand_idx[:int(len(rand_idx) * self.train_split)]
        val_idx = rand_idx[int(len(rand_idx) * self.train_split):]
        train_images = images[train_idx]
        print(train_images.max())
        print(train_images.min())
        train_labels = labels[train_idx]
        val_images = images[val_idx]
        val_labels = labels[val_idx]

        train_mean = train_images.mean()
        train_std = train_images.std()
        train_images = (train_images - train_mean) / train_std
        val_images  = (val_images - train_mean) / train_std
        train_label_mean = np.nanmean(train_labels, 0)
        train_label_std = np.nanmean(train_labels, 0)
        train_labels = (train_labels - train_label_mean) / train_label_std
        val_labels = (val_labels - train_label_mean) / train_label_std
        train_labels[np.isnan(train_labels)] = -999.
        val_labels[np.isnan(val_labels)] = -999.

        # Build CV dict
        cv_files, cv_labels = {}, {}
        cv_files[self.folds['train']] = train_images.astype(np.float32)
        cv_files[self.folds['val']] = val_images.astype(np.float32)
        cv_labels[self.folds['train']] = train_labels.astype(np.float32)
        cv_labels[self.folds['val']] = val_labels.astype(np.float32)
        return cv_files, cv_labels


import os
import numpy as np
import tensorflow as tf
from config import Config
from ops import tf_fun


class data_processing(object):
    def __init__(self):
        self.name = 'shape_connectome_baseline_200'
        self.output_name = 'shape_connectome_baseline_200'
        self.data_name = 'baseline'
        self.img_dir = 'imgs'
        self.contour_dir = '/media/data_cifs/synth/synth_connectomics/'
        self.im_extension = '.png'
        self.label_regex = r'(?<=length)\d+'
        self.config = Config()
        self.im_size = [350, 350]  # 600, 600
        self.model_input_image_size = [320, 320, 1]  # [107, 160, 3]
        self.nhot_size = [26]
        self.max_ims = 400
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
        self.preprocess = ['resize']  # ['resize_nn']
        self.meta = os.path.join('metadata', 'combined.npy')
        self.folds = {
            'train': 'train',
            'val': 'val'
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
        positive_meta = np.load(
            os.path.join(
                self.contour_dir,
                self.data_name,
                self.meta))
        positive_meta = positive_meta.reshape(-1, 7)
        ims, labs = self.list_files(
            positive_meta, self.data_name, cat=0)
        # labs = self.list_files(positive_meta, self.data_name, cat=1)

        if self.max_ims:
            all_ims = ims[:self.max_ims]
            all_labels = labs[:self.max_ims]
        else:
            all_ims = ims
            all_labels = labs
        num_ims = len(all_ims)

        # Balance CV folds
        np.random.seed(1)
        shuffle_idx = np.random.permutation(num_ims)
        shuffle_images = all_ims[shuffle_idx]
        shuffle_labels = all_labels[shuffle_idx]
        train_ims = shuffle_images[:self.train_size]
        train_labels = shuffle_labels[:self.train_size]
        val_ims = shuffle_images[200:]
        val_labels = shuffle_labels[200:]

        # Build CV dict
        cv_files, cv_labels = {}, {}
        cv_files[self.folds['train']] = train_ims
        cv_files[self.folds['val']] = val_ims
        cv_labels[self.folds['train']] = train_labels
        cv_labels[self.folds['val']] = val_labels
        return cv_files, cv_labels

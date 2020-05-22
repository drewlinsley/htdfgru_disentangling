import os
import cv2
import numpy as np
import tensorflow as tf
from config import Config
from ops import tf_fun
from glob import glob
from tqdm import tqdm
try:
    import pandas as pd
except Exception:
    print('Failed to import pandas.')


class data_processing(object):
    def __init__(self):
        self.name = 'new_LMD'
        self.output_name = 'new_LMD'
        self.kras_dir = '/media/data_cifs/andreas/pathology/2018-04-26/mar2019/LMD/patch_npys'
        self.im_extension = '.npy'
        self.label_regex = r'(?<=length)\d+'
        self.config = Config()
        self.im_size = [230, 230, 3]  # 600, 600
        self.model_input_image_size = [200, 200, 3]  # [107, 160, 3]
        self.output_size = [1]
        self.label_size = self.output_size
        self.default_loss_function = 'bce'
        self.score_metric = 'accuracy'
        self.store_z = False
        self.normalize_im = False
        self.all_flips = True
        self.balance = True
        self.shuffle = True
        self.input_normalization = 'none'  # 'zscore'
        self.preprocess = ['rgba2rgb', 'crop_center']  # ['resize_nn']
        self.LMD = ['3361805']
        self.val_set = self.LMD  # self.non_lung_cases_new  # self.non_lung_kras_cases_2017 + self.non_lung_non_kras_cases_2017  #  + self.non_lung_cases_new 
        self.folds = {
            'train': 'train',
            'val': 'val'
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

        all_ims = glob(
            os.path.join(
                self.kras_dir,
                '*%s' % self.im_extension))
        all_ims = np.array(all_ims)
        all_labels = np.array(
            [1
                if x.split(os.path.sep)[-1].split('_')[0] == 'KRAS'
                else 0 for x in all_ims])
        assert len(all_ims) == len(all_labels)

        # Split into CV sets
        val_idx = np.zeros(len(all_ims), dtype=bool)
        for m in tqdm(
                self.val_set,
                desc='Processing the validation index',
                total=len(self.val_set)):
            matches = np.array([True if m in x else False for x in all_ims])
            val_idx += matches
        train_ims = all_ims[~val_idx]
        train_labels = all_labels[~val_idx]
        val_ims = all_ims[val_idx]
        val_labels = all_labels[val_idx]

        # Balance +/- train sizes
        print 'Began with %s train images and %s val images' % (
            len(train_ims), len(val_ims))
        pos_examples = train_labels.sum()
        imbalance = pos_examples - (len(train_labels) - pos_examples)
        if imbalance > 0:
            neg_files = train_ims[train_labels == 0]
            neg_labels = train_labels[train_labels == 0]
            train_ims = np.concatenate((train_ims, neg_files))
            train_labels = np.concatenate((train_labels, neg_labels))
        else:
            pos_files = train_ims[train_labels == 1]
            pos_labels = train_labels[train_labels == 1]
            train_ims = np.concatenate((train_ims, pos_files))
            train_labels = np.concatenate((train_labels, pos_labels))
        print 'Balanced the train set to %s images' % len(train_ims)
        if self.shuffle:
            def shuffle_set(ims, labels):
                """Apply random shuffle to ims and labels."""
                rand_idx = np.random.permutation(len(ims))
                assert len(ims) == len(labels)
                ims = ims[rand_idx]
                labels = labels[rand_idx]
                return ims, labels
            train_ims, train_labels = shuffle_set(train_ims, train_labels)
            val_ims, val_labels = shuffle_set(val_ims, val_labels)

        # Build CV dict
        cv_files, cv_labels = {}, {}
        cv_files[self.folds['train']] = train_ims
        cv_files[self.folds['val']] = val_ims
        cv_labels[self.folds['train']] = train_labels
        cv_labels[self.folds['val']] = val_labels
        return cv_files, cv_labels

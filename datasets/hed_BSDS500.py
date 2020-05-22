import os
import numpy as np
import tensorflow as tf
import pandas as pd
from glob import glob
from config import Config
from ops import tf_fun
from utils import py_utils
from scipy import io, misc
from tqdm import tqdm


class data_processing(object):
    def __init__(self):
        self.name = 'hed_BSDS500'
        self.output_name = 'hed_BSDS500'
        self.images_dir = '/media/data_cifs/image_datasets/hed_bsds/HED-BSDS'
        self.processed_labels = 'processed_labels'
        self.processed_images = 'processed_images'
        self.config = Config()
        self.im_size = [321, 481, 3]
        # self.model_input_image_size = [196, 196, 3]
        self.model_input_image_size = [320, 320, 3]  # [224, 224, 3]
        self.val_model_input_image_size = [320, 320, 3]
        self.output_size = [321, 481, 1]
        self.label_size = self.output_size
        self.default_loss_function = 'pearson'
        self.score_metric = 'sigmoid_accuracy'
        self.aux_scores = ['f1']
        self.store_z = True
        self.input_normalization = 'none'  # 'zscore'
        self.preprocess = [None]  # Preprocessing before tfrecords
        self.folds = {
            'train': 'train',
            'val': 'val',
            'test': 'test',
        }
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
        train_files = pd.read_csv(
            os.path.join(
                self.images_dir, 'train_pair.lst'), header=None, delim_whitespace=True)
        test_files = pd.read_csv(
            os.path.join(
                self.images_dir, 'test.lst'), header=None, delim_whitespace=True)
        val_names = glob(
            os.path.join('/media/data_cifs/image_datasets/BSDS500/images/val', '*.jpg')) 
        val_names = [v.split(os.path.sep)[-1].split('.')[0] for v in val_names]

        # Split off val set
        train_rows, val_rows = [], []
        for idx in range(len(train_files)):
            row = train_files.iloc[idx]
            split_name = row[0].split(os.path.sep)[-1].split('.')[0]
            try:
                val_idx = val_names.index(split_name)
                val_names.pop(val_idx)
                val_rows += [row]
            except Exception:
                train_rows += [row]
        train_files = pd.concat(train_rows, 1).transpose()
        val_files = pd.concat(val_rows, 1).transpose()
        train_ims = [os.path.join(self.images_dir, x) for x in train_files[0]]
        train_labs = [os.path.join(self.images_dir, x) for x in train_files[1]]
        val_ims = [os.path.join(self.images_dir, x) for x in val_files[0]]
        val_labs = [os.path.join(self.images_dir, x) for x in val_files[1]]
        test_ims = [os.path.join(self.images_dir, x) for x in test_files[0]]
        test_labs = test_ims
        cv_files, cv_labels = {}, {}
        cv_files[self.folds['train']] = train_ims
        cv_files[self.folds['val']] = val_ims
        cv_files[self.folds['test']] = test_ims
        cv_labels[self.folds['train']] = train_labs
        cv_labels[self.folds['val']] = val_labs
        cv_labels[self.folds['test']] = test_labs
        return cv_files, cv_labels

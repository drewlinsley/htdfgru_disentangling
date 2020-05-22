import os
import numpy as np
import pandas as pd
import tensorflow as tf
from glob import glob
from config import Config
from ops import tf_fun
from utils import py_utils
from scipy import io, misc
from skimage.io import imread
from tqdm import tqdm
from utils import image_processing as im_proc


class data_processing(object):
    def __init__(self):
        self.output_name = 'BSDS500_100_hed'
        self.im_extension = '.jpg'
        self.lab_extension = '.mat'
        self.images_dir = '/media/data_cifs/pytorch_projects/datasets/BSDS500_crops/data/images/train'
        self.val_images_dir = '/media/data_cifs/pytorch_projects/datasets/BSDS500_crops/data/images/val'
        self.processed_labels = 'processed_labels'
        self.processed_images = 'processed_images'
        self.config = Config()
        self.train_size = int(1000 * 1)
        self.im_size = [320, 320, 3]  # [321, 481, 3]
        self.model_input_image_size = [320, 320, 3]  # [224, 224, 3]
        self.val_model_input_image_size = [320, 320, 3]
        self.output_size = [320, 320, 1]  # [321, 481, 1]
        self.label_size = self.output_size
        self.default_loss_function = 'pearson'
        self.score_metric = 'sigmoid_accuracy'
        self.aux_scores = ['f1']
        self.store_z = True
        self.input_normalization = 'none'  # 'zscore'
        self.preprocess = [None]  # Preprocessing before tfrecords
        self.folds = {
            'train': 'train',
            'val': 'val'
        }
        self.fold_options = {
            'train': 'mean',
            'val': 'mean'
        }
        self.targets = {
            'image': tf_fun.bytes_feature,
            'label': tf_fun.bytes_feature,
            'height': tf_fun.int64_feature,
            'width': tf_fun.int64_feature,
        }
        self.tf_dict = {
            'image': tf_fun.fixed_len_feature(dtype='string'),
            'label': tf_fun.fixed_len_feature(dtype='string'),
            'height': tf_fun.fixed_len_feature(dtype='int64'),
            'width': tf_fun.fixed_len_feature(dtype='int64'),
        }
        self.tf_reader = {
            'image': {
                'dtype': tf.float32,
                'reshape': self.im_size
            },
            'label': {
                'dtype': tf.float32,
                'reshape': self.output_size
            },
            'height': {
                'dtype': tf.int64,
                'reshape': []
            },
            'width': {
                'dtype': tf.int64,
                'reshape': []
            },
        }

    def get_data(self):
        files, labels = self.get_files()
        return files, labels

    def get_files(self):
        """Get the names of files."""
        train_images = glob(os.path.join(self.images_dir, '*%s' % self.im_extension)) 
        train_labels = [x.replace(self.im_extension, '.npy').replace('images', 'groundTruth') for x in train_images]
        val_images = glob(os.path.join(self.val_images_dir, '*%s' % self.im_extension))
        val_labels = [x.replace(self.im_extension, '.npy').replace('images', 'groundTruth') for x in val_images]
        val_images = np.array(val_images)
        val_labels = np.array(val_labels)
        
        # Get HED images
        np.random.seed(0)
        bsds_path = '/media/data_cifs/image_datasets/hed_bsds/HED-BSDS'
        f = os.path.join(bsds_path, 'train_pair.lst')
        train_paths = pd.read_csv(f, header=None, delimiter=' ')
        train_images = train_paths[0].values
        train_labels = train_paths[1].values
        shuffle_idx = np.random.permutation(len(train_images))
        train_images = train_images[shuffle_idx]
        train_labels = train_labels[shuffle_idx]

        # Read the HED images
        ims, labs = [], []
        r_stride = self.model_input_image_size[0]
        for im, lab in tqdm(zip(train_images, train_labels), desc='Loading HEDs', total=len(train_images)):
            lim = imread(os.path.join(bsds_path, im))
            llab = imread(os.path.join(bsds_path, lab)) // 255.
            if len(llab.shape) == 3:
                llab = llab[..., 0]
            lsh = lim.shape
            if lsh[0] > lsh[1]:
                # Flip all to landscape
                lim = lim.transpose((1, 0, 2))
                llab = llab.transpose((1, 0))
                lsh = lim.shape
            llab = llab[..., None]
             
            if lsh[0] < r_stride:
                # Pad to 320
                up_offset = (r_stride - lsh[0]) // 2
                down_offset = up_offset
                if up_offset + down_offset + lsh[0] < r_stride:
                    down_offset += 1
                elif up_offset + down_offset + lsh[0] > r_stride:
                    down_offset -= 1
                pad_up_offset = np.zeros((up_offset, lsh[1], 3))
                pad_down_offset = np.zeros((down_offset, lsh[1], 3))
                lim = np.concatenate((pad_up_offset, lim, pad_down_offset), 0)
                pad_up_offset = np.zeros((up_offset, lsh[1], 1)) - 1
                pad_down_offset = np.zeros((down_offset, lsh[1], 1)) - 1
                llab = np.concatenate((pad_up_offset, llab, pad_down_offset), 0)
            elif lsh[1] < r_stride:
                # Pad to 320
                up_offset = (r_stride - lsh[1]) // 2
                down_offset = up_offset 
                if up_offset + down_offset + lsh[1] < r_stride:
                    down_offset += 1
                elif up_offset + down_offset + lsh[1] > r_stride:
                    down_offset -= 1
                pad_up_offset = np.zeros((lsh[0], up_offset, 3))
                pad_down_offset = np.zeros((lsh[0], down_offset, 3))
                lim = np.concatenate((pad_up_offset, lim, pad_down_offset), 1)
                up_offset = np.zeros((lsh[0], up_offset, 1)) - 1
                pad_down_offset = np.zeros((lsh[0], down_offset, 1)) - 1
                llab = np.concatenate((pad_up_offset, llab, pad_down_offset), 1)

            ims += [lim]
            labs += [llab]
            # # Make crops
            # lsh = lim.shape
            # r_crop_idx = np.arange(0, lsh[0], rstride)
            # c_crop_idx = np.arange(0, lsh[1], rstride)
            # for ro in r_crop_idx:
            #     for co in c_crop_idx:
            #         ims += [


        # Build CV dict
        cv_files, cv_labels = {}, {}
        cv_files[self.folds['train']] = train_images
        cv_files[self.folds['val']] = val_images
        cv_labels[self.folds['train']] = train_labels
        cv_labels[self.folds['val']] = val_labels
        return cv_files, cv_labels


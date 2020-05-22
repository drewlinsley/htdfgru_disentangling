import os
import numpy as np
import tensorflow as tf
from config import Config
from ops import tf_fun
# from utils import image_processing
from glob import glob
from tqdm import tqdm
# import cv2
try:
    import imageio as io
except Exception:
    print('Failed to import imageio.')
try:
    import pandas as pd
except Exception:
    print('Failed to import pandas.')


class data_processing(object):
    def __init__(self):
        self.name = 'cube_plus'
        self.output_name = 'cube_plus'
        self.main_dir = '/media/data_cifs/image_datasets/cube_plus'
        self.image_dir = os.path.join(self.main_dir, 'images')
        self.label_file = os.path.join(self.main_dir, 'cube+_gt.txt')
        self.im_extension = '.PNG'
        self.label_regex = r'(?<=length)\d+'
        self.config = Config()
        self.im_size = [1732, 2601, 3]  # 600, 600
        self.model_input_image_size = [288, 433, 3]  # [107, 160, 3]
        self.max_ims = 125000
        self.output_size = [3]
        self.label_size = self.output_size
        self.default_loss_function = 'bce'
        self.score_metric = 'accuracy'
        self.store_z = False
        self.normalize_im = False
        self.all_flips = True
        self.balance = True
        self.shuffle = True
        self.calculate_moments = True
        self.input_normalization = 'none'  # 'zscore'
        self.preprocess = []
        self.folds = {
            'train': 'train',
            'val': 'val'
        }
        self.cv_split = 0.1
        self.cv_balance = True
        self.targets = {
            'image': tf_fun.bytes_feature,
            'label': tf_fun.float_feature
        }
        self.tf_dict = {
            'image': tf_fun.fixed_len_feature(dtype='string'),
            'label': tf_fun.fixed_len_feature(
                dtype='float32',
                length=self.output_size[0])
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

    def cube_plus_process(self, image, black_level=2048, process=True):
        """Apply cube plus preprocessing."""
        if process:
            saturation = image.max() - 2
            image -= black_level
            image[image < 0] = 0
            thresh = saturation - black_level
            m = (image >= thresh).astype(np.float32).sum(-1) > 0
            m[1049:, 2049:] = True
            m = m[..., None].repeat(3, axis=-1)
            image[m] = 0
        else:
            image[1049:, 2049:] = 0
        return image

    def get_data(self):
        """Get the names of files."""
        all_ims = glob(
            os.path.join(
                self.image_dir,
                '*%s' % self.im_extension))
        labels = np.array(
            [x.tolist()[0].split(' ') for x in pd.read_csv(
                self.label_file, header=None).as_matrix()]).astype(
            np.float32)
        image_data = np.zeros(
            [len(all_ims)] + self.im_size, dtype=np.float32)
        for idx, f in tqdm(
                enumerate(all_ims),
                total=len(all_ims),
                desc='Loading images'):
            im = io.imread(f, format='PNG-FI')
            image_data[idx] = self.cube_plus_process(im, process=True)

        # Print out the max value
        print('*' * 20)
        print('Max value: %s' % image_data.max())
        print('*' * 20)

        # Split into CV folds
        num_val = int(self.cv_split * len(labels))
        val_idx = np.random.permutation(len(labels))[:num_val]
        cv_idx = np.zeros(len(labels), dtype=bool)
        cv_idx[val_idx] = True
        train_ims = image_data[~cv_idx]
        train_labels = labels[~cv_idx]
        val_ims = image_data[cv_idx]
        val_labels = labels[cv_idx]

        # Build CV dict
        cv_files, cv_labels = {}, {}
        cv_files[self.folds['train']] = train_ims
        cv_files[self.folds['val']] = val_ims
        cv_labels[self.folds['train']] = train_labels
        cv_labels[self.folds['val']] = val_labels
        return cv_files, cv_labels

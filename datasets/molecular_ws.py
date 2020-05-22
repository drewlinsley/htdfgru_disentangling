import os
import numpy as np
import tensorflow as tf
from config import Config
from ops import tf_fun
from utils import image_processing
from glob import glob
from tqdm import tqdm
try:
    import pandas as pd
except Exception:
    print('Failed to import pandas.')


class data_processing(object):
    def __init__(self):
        self.name = 'molecular_ws'
        self.output_name = 'molecular_ws'
        self.kras_dir = '/media/data_cifs/andreas/pathology/2018-04-26/mar2019/data-recreation/4-07-19/256_npys'
        self.im_extension = '.npy'
        self.label_regex = r'(?<=length)\d+'
        self.config = Config()
        self.im_size = [256, 256, 3]  # 600, 600
        self.model_input_image_size = [224, 224, 3]  # [107, 160, 3]
        self.max_ims = 125000
        self.output_size = [1]
        self.label_size = self.output_size
        self.default_loss_function = 'cce'
        self.score_metric = 'accuracy'
        self.store_z = False
        self.normalize_im = False
        self.all_flips = True
        self.balance = True
        self.shuffle = True
        self.calculate_moments = False
        self.input_normalization = 'none'  # 'zscore'
        self.preprocess = ['rgba2rgb']  # ['to_float32', 'crop_center']  # , 'exclude_white']  # 'rgba2rgb', 
        self.LMD = ['3361805', '3361851', '3361808', '3361848', '3361832', '3361802', '3361807', '3361871', '3361894', '3361925', '3361830']
        self.non_lung_kras_cases_2017 = ['Case-6', 'Case-14']
        self.non_lung_non_kras_cases_2017 = ['Case-1', 'Case-2', 'Case-3', 'Case-4', 'Case-6', 'Case-7', 'Case-11', 'Case-12', 'Case-14']
        self.non_lung_cases_new = ['3361882', '3361883', '3361845', '3361846', '3361847', '3361935', '3361809', '3361810', '3361815', '3361820', '3361821', '3361868', '3361890', '3361869', '3361870', '3361901', '3361936', '3361942', '3361945', '3361817', '3361957', '3361966', '3361967', '3361823', '3361840', '3361885', '3361886', '3361887', '3361891', '3361892', '3361911', '3361940', '3361941', '3361959', '3361960', '3361969']
        self.val_set = self.LMD  # self.non_lung_cases_new  # self.non_lung_kras_cases_2017 + self.non_lung_non_kras_cases_2017  #  + self.non_lung_cases_new 
        self.val_set = self.non_lung_kras_cases_2017 + self.non_lung_non_kras_cases_2017 + self.non_lung_cases_new 
        self.folds = {
            'train': 'train',
            'val': 'val'
        }
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
        assert len(all_ims), 'No images found'
        all_ims = np.array(all_ims)
        all_im_names = np.copy(all_ims)
        all_labels = np.array(
            [1
                if 'KRAS' in x.split(os.path.sep)[-1]
                else 0 for x in all_ims])
        assert len(all_ims) == len(all_labels)

        # Load the images and then normalize per-slide
        if self.calculate_moments:
            all_slides = np.array(
                [x.split(os.path.sep)[-1].split('_')[1] for x in all_ims])
            means, stds = {}, {}
            unique_slides = np.unique(all_slides)

            # First load the images
            image_data = np.zeros(
                [len(all_ims)] + self.im_size, dtype=np.float32)
            exclude_idx = np.ones(len(all_ims)).astype(bool)
            hw = np.prod(self.im_size[:-1])
            for idx, f in tqdm(
                    enumerate(all_ims),
                    total=len(all_ims),
                    desc='Loading images'):
                image_data[idx] = image_processing.crop_center(
                    np.load(f)[:, :, :-1], self.im_size).astype(np.float32)
                thresh = 0.25
                white_check = np.sum(
                    np.std(image_data[idx], axis=-1) < 0.01) / hw
                if white_check > thresh:
                    exclude_idx[idx] = False
            all_slides = all_slides[exclude_idx]
            all_ims = all_ims[exclude_idx]
            all_labels = all_labels[exclude_idx]
            image_data = image_data[exclude_idx]
            for slide_idx in tqdm(
                    unique_slides,
                    desc='Per-slide moments',
                    total=len(unique_slides)):
                idx = all_slides == slide_idx
                it_files = image_data[idx]
                im_stack = np.zeros(
                    [len(it_files)] + self.im_size, dtype=np.float32)
                for fidx, f in tqdm(enumerate(it_files), total=len(it_files)):
                    try:
                        im_stack[fidx, :, :] = f
                    except Exception:
                        print 'Failed to load image'
                means[slide_idx] = np.mean(im_stack.astype(np.float32))
                stds[slide_idx] = np.std(im_stack.astype(np.float32))

            for slide_idx in tqdm(
                    unique_slides,
                    desc='Applying moments',
                    total=len(unique_slides)):
                idx = all_slides == slide_idx
                image_data[idx] = (
                    image_data[idx] - means[slide_idx]) / stds[slide_idx]
            all_ims = image_data

        # Split into CV sets
        val_idx = np.zeros(len(all_im_names), dtype=bool)
        ns = []
        for idx, m in tqdm(
                enumerate(all_im_names),
                desc='Processing the validation index',
                total=len(all_im_names)):
            matches = []
            f = m.split(os.path.sep)[-1].split('_')[0]
            ns += [f]
            if f in self.val_set:
                val_idx[idx] = True 
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

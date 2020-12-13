import os
import numpy as np
import tensorflow as tf
from config import Config
from ops import tf_fun
from scipy import misc
from skimage.transform import resize
from glob import glob


class data_processing(object):
    def __init__(self):
        self.name = 'orientation_probe'
        # self.name = 'plaid_surround'
        self.output_name = 'orientation_probe_151_viz'
        self.img_dir = 'imgs'
        # self.contour_dir = '/media/data_cifs/cluster_projects/refactor_gammanet/plaid_surround'
        self.contour_dir = '/media/data_cifs_lrs/projects/prj_neural_circuits/refactor_gammanet/{}'.format(self.name)
        # self.perturb_a = "perturb_viz/gammanet_full_plaid_surround_outputs_data.npy"
        self.perturb_a = "perturb_viz/bsds_tcs.npy"
        self.perturb_b = "perturb_viz/0.npz"  # bsds_tcs.npy"

        self.im_extension = '.png'
        self.label_regex = r'(?<=length)\d+'
        self.config = Config()
        self.im_size = [480, 320, 3]  # [500, 500]  # 600, 600
        self.model_input_image_size = [480, 320, 3]  # [107, 160, 3]
        # self.output_size = [112, 112, 128]  # [320, 480, 1]  # [321, 481, 1]
        self.output_size = [1, 2048 + 6]  # [320, 480, 1]  # [321, 481, 1]
        self.max_ims = 100000
        self.label_size = self.output_size
        self.default_loss_function = 'l2'
        self.store_z = False
        self.normalize_im = False
        self.all_flips = True
        self.shuffle = True
        self.input_normalization = 'none'  # 'zscore'
        self.preprocess = []  # ['resize']  # ['resize_nn']
        self.meta = os.path.join('metadata', '1.npy')
        self.folds = {
            'train': 'train',
            'val': 'val',
            'test': 'test',
        }
        self.cv_split = 0.1
        self.targets = {
            'image': tf_fun.bytes_feature,
            'label': tf_fun.float_feature
        }
        self.tf_dict = {
            'image': tf_fun.fixed_len_feature(dtype='string'),
            'label': tf_fun.fixed_len_feature(
                dtype='float32',
                length=self.output_size)
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
        files, labs = [], []
        for idx, f in enumerate(meta):
            files += [
                os.path.join(
                    self.contour_dir,
                    directory,
                    f[0],
                    f[1])]
            deg = float(f[4])
            labs += [[np.sin(deg * np.pi / 180.), np.cos(deg * np.pi / 180.)]]
        return np.asarray(files), np.asarray(labs)

    def get_data(self):
        """Get the names of files."""
        test_images = np.array(glob('/media/data_cifs_lrs/pytorch_projects/datasets/BSDS500_crops/data/images/test_nocrop/*.jpg'))
        test_labels = np.array([x.replace('images', 'groundTruth').replace('.jpg', '.npy') for x in test_images])
        test_labels = np.array([np.load(x) for x in test_labels])
        keep_idx = np.array([True if x.shape[0] > x.shape[1] else False for x in test_labels])
        test_images = test_images[keep_idx]
        test_labels = test_labels[keep_idx]
        test_images = np.stack([misc.imread(x) for x in test_images], 0)
        test_labels = np.stack(test_labels, 0)

        test_images = test_images[:, :480, :320]
        test_labels = test_labels[:, :480, :320][..., None]

        # Pascal
        test_images = test_images - np.asarray([123.68, 116.78, 103.94])[None, None]

        # Logits for perturbation
        logits_b = np.load(self.perturb_a, allow_pickle=True)
        logits_b = logits_b.squeeze()[109, 119].reshape(1, -1)  # T
        print("Extracting tuning curves for targets.")

        # Logits for target
        lb = np.load(self.perturb_b, allow_pickle=True)
        logits_a = lb["act"].squeeze()
        del lb.f
        lb.close()
        logits_a = logits_a[109: 109 + 4, 119: 119 + 4].reshape(1, -1)
        print("Extracting tuning curves for targets.")

        # Combined logits
        logits = np.concatenate((logits_a, logits_b), 1)

        # Build CV dict
        cv_files, cv_labels = {}, {}
        # test_images = test_images[0][None]
        # logits = logits[0][None]
        cv_files[self.folds['train']] = test_images[0][None]
        cv_files[self.folds['val']] = test_images[0][None]
        cv_files[self.folds['test']] = test_images[0][None]
        cv_labels[self.folds['train']] = logits[0][None]
        cv_labels[self.folds['val']] = logits[0][None]  # val_labels
        cv_labels[self.folds['test']] = logits[0][None]  # test_labels
        return cv_files, cv_labels


import os
import numpy as np
import tensorflow as tf
from config import Config
from ops import tf_fun
from scipy import misc
from skimage.transform import resize


class data_processing(object):
    def __init__(self):
        self.name = 'orientation_probe'
        # self.name = 'plaid_surround'
        self.output_name = 'orientation_probe_1_viz'
        self.img_dir = 'imgs'
        # self.contour_dir = '/media/data_cifs/cluster_projects/refactor_gammanet/plaid_surround'
        self.contour_dir = '/media/data_cifs_lrs/projects/prj_neural_circuits/refactor_gammanet/{}'.format(self.name)
        # self.perturb_a = "perturb_viz/gammanet_full_plaid_surround_outputs_data.npy"
        self.perturb_a = "../refactor_gammanet/orientation_probe_outputs/BSDS_vgg_gratings_simple_orientation_test_2020_09_27_10_14_47_500413.npz"
        self.perturb_b = "perturb_viz/gammanet_full_orientation_probe_outputs_data.npy"

        self.im_extension = '.png'
        self.label_regex = r'(?<=length)\d+'
        self.config = Config()
        self.im_size = [224, 224, 3]  # [500, 500]  # 600, 600
        self.model_input_image_size = [224, 224, 3]  # [107, 160, 3]
        # self.output_size = [112, 112, 128]  # [320, 480, 1]  # [321, 481, 1]
        self.output_size = [1, 2048 + 6 + 6]  # [320, 480, 1]  # [321, 481, 1]
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
        meta_train = np.load(
            os.path.join(
                self.contour_dir,
                'test',
                self.meta), allow_pickle=True)
        meta_train = meta_train.reshape(-1, 12)
        all_ims, all_labs = self.list_files(
            meta_train, 'test')
        if self.max_ims:
            all_ims = all_ims[:self.max_ims]
            all_labs = all_labs[:self.max_ims]
        num_ims = len(all_ims)

        # Get test data
        meta_test = np.load(
            os.path.join(
                self.contour_dir,
                'test',
                self.meta), allow_pickle=True)
        meta_test = meta_test.reshape(-1, 12)
        test_ims, test_labs = self.list_files(
            meta_test, 'test')
        test_images = test_ims

        # Get images
        test_images = np.stack([misc.imread(x) for x in test_images], 0)

        # Modulate
        modulate = 10
        test_images /= modulate
        offset = (255 / 2) - ((255 / modulate) / 2)
        test_images += offset

        # Stack3d
        test_images = test_images[..., None].repeat(3, -1)

        # Resize
        res_images = np.zeros([test_images.shape[0]] + self.model_input_image_size, dtype=test_images.dtype)
        for idx, im in enumerate(test_images):
            res_images[idx] = resize(im, self.model_input_image_size[:-1], preserve_range=True)
        # test_images = resize(test_images, self.model_input_image_size, preserve_range=True)
        test_images = res_images

        # Pascal
        test_images = test_images - np.asarray([123.68, 116.78, 103.94])[None, None]

        # Get targets
        """
        logits = np.asarray([x["logits"] for x in targets])
        logits = logits.squeeze(1)
        """
        # Phase labels
        phase = np.load("perturb_viz/INSILICO_BSDS_vgg_gratings_simple_phase_outputs.npy").T
        if len(phase) == 181: phase = phase[1:]

        # Logits for perturbation
        logits_a = np.load(self.perturb_a)
        logits_a = logits_a["test_dict"]
        logits_a = np.asarray([x["ephys"] for x in logits_a])
        logits_a = logits_a.squeeze()  # T
        print("Extracting tuning curves for targets.")

        # Logits for target
        logits_b = np.load(self.perturb_b)
        logits_b = logits_b.T
        if len(logits_b) == 181:
            logits_b = logits_b[1:]
        logits_b = np.concatenate((logits_b, phase[0][None].repeat(180, 0)), -1)

        # Combined logits
        logits = np.concatenate((logits_a, logits_b), -1)

        # Build CV dict
        cv_files, cv_labels = {}, {}
        # test_images = test_images[0][None]
        # logits = logits[0][None]
        cv_files[self.folds['train']] = test_images[0][None]
        cv_files[self.folds['val']] = test_images
        cv_files[self.folds['test']] = test_images
        cv_labels[self.folds['train']] = logits[0][None]
        cv_labels[self.folds['val']] = logits  # val_labels
        cv_labels[self.folds['test']] = logits  # test_labels
        return cv_files, cv_labels


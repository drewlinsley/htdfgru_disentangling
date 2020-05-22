import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
from ops import tf_fun
from utils import py_utils
from tqdm import tqdm
from skimage import io


PASCAL = np.array([123.68, 116.78, 103.94][::-1])[None, None, :]


def main(f, tag):
    """Plot BSDS figs."""
    fn = f.split(os.path.sep)[-1].strip('.npz')
    dir_name = f.split(os.path.sep)[0]
    fig_dir = os.path.join(dir_name, 'trip_%s_%s' % (tag, fn))
    res_dir = os.path.join(dir_name, 'preds_%s_%s' % (tag, fn))
    py_utils.make_dir(fig_dir)
    py_utils.make_dir(res_dir)
    d = np.load(f)
    test_dict = d['test_dict']
    if 'portrait' in f:
        from datasets.BSDS500_test_portrait import data_processing as dp
    elif 'landscape' in f:
        from datasets.BSDS500_test_landscape import data_processing as dp
    elif 'multicue_edges' in f:
        from datasets.multicue_100_edges_jk_test import data_processing as dp
    else:
        raise NotImplementedError(f)
    dP = dp()
    files = dP.get_test_files()
    print('Saving files to %s and %s' % (fig_dir, res_dir))
    for idx, (td, im) in tqdm(
            enumerate(zip(test_dict, files)), total=len(test_dict)):
        f = plt.figure()
        im_name = im.split(os.path.sep)[-1].split('.')[0]
        score = tf_fun.sigmoid_fun(td['logits'].squeeze())
        proc_im = (td['images'].squeeze()[..., [2, 1, 0]] + PASCAL).astype(np.uint8)
        proc_lab = td['labels'].squeeze()
        sc_shape = score.shape
        # if sc_shape[0] > sc_shape[1]:
        #     diff_h = sc_shape[0] - 481
        #     diff_w = sc_shape[1] - 321
        #     score = score[diff_h // 2:-(diff_h - diff_h // 2), diff_w // 2: (-diff_w - diff_w // 2)]
        #     proc_im = proc_im[diff_h // 2:-(diff_h - diff_h // 2), diff_w // 2: (-diff_w - diff_w // 2)]  # [:481, :321]
        #     proc_lab = proc_lab[diff_h // 2:-(diff_h - diff_h // 2), diff_w // 2: (-diff_w - diff_w // 2)]  # [:481, :321]
        # elif sc_shape[1] > sc_shape[0]:
        #     diff_h = sc_shape[1] - 481
        #     diff_w = sc_shape[0] - 321
        #     score = score[diff_h // 2:-(diff_h - diff_h // 2), diff_w // 2: (-diff_w - diff_w // 2)]
        #     proc_im = proc_im[diff_h // 2:-(diff_h - diff_h // 2), diff_w // 2: (-diff_w - diff_w // 2)]  # [:481, :321]
        #     proc_lab = proc_lab[diff_h // 2:-(diff_h - diff_h // 2), diff_w // 2: (-diff_w - diff_w // 2)]  # [:481, :321]
        # else:
        #     raise RuntimeError(sc_shape)
        plt.subplot(131)
        plt.imshow(proc_im)
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(proc_lab)
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(score, cmap='Greys_r')
        # io.imsave(
        #     os.path.join(
        #         res_dir,
        #         '%s.tiff' % im_name),
        #     score)
        np.save(
            os.path.join(
                res_dir,
                '%s' % im_name),
            score)
        plt.savefig(
            os.path.join(
                fig_dir,
                '%s.pdf' % im_name), dpi=150)
        plt.close(f)


if __name__ == '__main__':
    """Flags are:

    experiment (STR): the name of a Parent experiment class you are using.
        model (STR): the name of a single model you want to
            train/test (overwrites parent experiment params)
        train (STR): a dataset class you want to use for
            training (defaults to tfrecords;
            overwrites parent experiment params)
        val (STR): see above
        num_batches (int): overwrite the experiment default
            # of validation_steps per validation
    ckpt (STR): the full path to a model checkpoint you will
        restore "model" with.
    reduction (int): DEPRECIATED reduce dataset size in training by a factor
    out_dir (STR): custom directory name to store your val output
    gpu (STR): gpu name for scoping
    cpu (STR): cpu name for scoping
    add_config (STR): add a string to your model-specific saved config file
    map_out (STR): only used if mAP is requested in experiment
        custom output folder.
    transfer (BOOL): DEPRECIATED custom transfer learning approach
    placeholders (BOOL): Use placeholders in training/val.
    test (BOOL): Use dataset-class test routine. Saves a npz with test data.
    no_db (BOOL): Do not use database functions.
    no_npz (BOOL): Does not save npz with test data. Only for test data!
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--f',
        dest='f',
        type=str,
        default=None,
        help='File path.')
    parser.add_argument(
        '--tag',
        dest='tag',
        type=str,
        default=None,
        help='Experiment tag.')
    args = parser.parse_args()
    main(**vars(args))

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from utils import py_utils


def sigmoid_fun(x):
    """Apply sigmoid to maps before mAP."""
    return 1 / (1 + np.exp(x))


df = pd.read_csv(
    os.path.join(
        'data_to_process_for_jk',
        'generalize_snemi_experiment_data',
        'generalize_snemi_experiment_data_grouped.csv'))
sel, subsel = 0, 3

out_dir = os.path.join('data_to_process_for_jk', 'generalization_membranes')
py_utils.make_dir(out_dir)
datasets = np.unique(df.val_dataset)
models = np.unique(df.model)
for d in tqdm(
        datasets,
        desc='Cycling through datasets',
        total=len(datasets)):
    f = plt.figure()
    plt.suptitle(d)
    for idx, m in enumerate(models):
        data = np.load(
            df[np.logical_and(
                df.model == m, df.val_dataset == d)].file_name.values[0])
        data = data['val_dict'][sel]
        logs = data['logits']
        if idx == 0:
            ims = data['images']
            labs = data['labels']
            plt.subplot(141)
            plt.axis('off')
            plt.imshow(ims[subsel].squeeze())
            plt.subplot(142)
            plt.axis('off')
            plt.imshow(labs[subsel].squeeze())
            plt.subplot(143)
            plt.axis('off')
            plt.title(m)
            plt.imshow(sigmoid_fun(logs[subsel, ..., 0]).squeeze())
        else:
            plt.subplot(144)
            plt.axis('off')
            plt.title(m)
            plt.imshow(sigmoid_fun(logs[subsel, ..., 0]).squeeze())
    plt.savefig(os.path.join(out_dir, '%s.pdf' % d))
    plt.close(f)

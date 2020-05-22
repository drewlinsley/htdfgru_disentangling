import os
import numpy as np
import pandas as pd
from glob import glob
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns
from utils import py_utils


def gather_data(data_files, keep_models, generalization=False):
    """Gather data from npzs in data_files."""
    # Start data proc loop
    # val_images, val_labels, val_logits = [], [], []
    # model_names, train_datasets = [], []
    dm = []
    for idx, f in tqdm(
            enumerate(data_files),
            total=len(data_files),
            desc='Gathering data for JK'):
        try:
            d = np.load(f)
            config = d['config'].item()
        except Exception as e:
            print 'Failed on %s with %s' % (f, e)
        if check_model(config.model, keep_models):
            continue
        elif generalization:
            if 'snemi' in config.train_dataset or \
                    'cremi' in config.train_dataset or \
                    'fib' in config.train_dataset or \
                    'berson' in config.train_dataset:
                srch = glob(
                    os.path.join(
                        '/media/data_cifs/cluttered_nist_experiments/checkpoints',
                        '%s' % f.split(os.path.sep)[-1].split('.')[0],
                        '*.ckpt*.meta'))
                if len(srch):
                    ckpt = srch[0].split('.meta')[0]
                else:
                    ckpt = False
                maps = d['maps']
                map_len = 1  # len(maps)
                dm += [np.vstack([
                    # [idx] * map_len,
                    np.mean(maps),
                    [config.train_dataset] * map_len,
                    [config.val_dataset] * map_len,
                    [f] * map_len,
                    [ckpt] * map_len,
                    [config.model] * map_len
                ])]
        elif 'snemi' in config.train_dataset:
            srch = glob(
                os.path.join(
                    '/media/data_cifs/cluttered_nist_experiments/checkpoints',
                    '%s' % f.split(os.path.sep)[-1].split('.')[0],
                    '*.ckpt*.meta'))
            if len(srch):
                ckpt = srch[0].split('.meta')[0]
            else:
                ckpt = False
            maps = d['maps']
            map_len = 1  # len(maps)
            dm += [np.vstack([
                # [idx] * map_len,
                np.mean(maps),
                [config.train_dataset] * map_len,
                [config.val_dataset] * map_len,
                [f] * map_len,
                [ckpt] * map_len,
                [config.model] * map_len
            ])]
    df = pd.DataFrame(
        np.concatenate(dm, axis=1).transpose(),
        columns=[
            # 'model_idx',
            'average_precision',
            'train_dataset',
            'val_dataset',
            'file_name',
            'ckpt',
            'model'])
    return df, dm


def check_model(model_name, model_list):
    """Return False if model_name in model_list."""
    for m in model_list:
        if model_name == m:
            return False
    return True


def process_data(df, out_name, out_data, generalization=False, max_samples=10):
    """Prepare data for saving to csv + plotting."""
    # Process df so that no model has > max_samples entries
    unique_train = np.unique(df.train_dataset)
    unique_val = np.unique(df.val_dataset)
    unique_model = np.unique(df.model)
    proc_df = []
    if generalization:
        for va in unique_val:
            for mo in unique_model:
                idx = np.logical_and(
                    df.val_dataset == va,
                    df.model == mo)
                it_df = df[idx]
                it_df = it_df.sort_values(
                    'average_precision', ascending=False)
                it_mask = np.ones(len(it_df.file_name), dtype=bool)
                it_mask[max_samples:] = False
                proc_df += [it_df]
    else:
        for tr in unique_train:
            for va in unique_val:
                for mo in unique_model:
                    idx = np.logical_and(
                        np.logical_and(
                            df.train_dataset == tr, df.val_dataset == va),
                        df.model == mo)
                    it_df = df[idx]
                    it_df = it_df.sort_values(
                        'average_precision', ascending=False)
                    it_mask = np.ones(len(it_df.file_name), dtype=bool)
                    it_mask[max_samples:] = False
                    proc_df += [it_df]
    df = pd.concat(proc_df)
    df[df.train_dataset == 'snemi_combos_200'] = 'snemi_200'
    if generalization:
        df.train_dataset[df.train_dataset == 'snemi_200_test'] = 'snemi_200'
        df.val_dataset[df.val_dataset == 'snemi_200_test'] = 'snemi_200'

    # Group data
    grouped_data = df.groupby(
        ['train_dataset', 'val_dataset', 'model']).agg('max').reset_index()

    # Ckpt data
    ckpts_data = df[df.ckpt.str.contains('/media')].groupby(
        ['train_dataset', 'val_dataset', 'model', 'ckpt']).agg(
        'max').reset_index()

    # Save csvs
    df.to_csv(os.path.join(out_data, '%s_raw.csv' % out_name))
    grouped_data.to_csv(os.path.join(out_data, '%s_grouped.csv' % out_name))
    if generalization:
        pass
    else:
        ckpts_data.to_csv(os.path.join(out_data, '%s_ckpts.csv' % out_name))
        ckpts_data['model'].to_csv(
            os.path.join(out_data, '%s_models_only.csv' % out_name))
        ckpts_data['ckpt'].to_csv(
            os.path.join(out_data, '%s_ckpts_only.csv' % out_name))
        ckpts_data['train_dataset'].to_csv(
            os.path.join(out_data, '%s_trainds_only.csv' % out_name))

    # Recode the datasets
    # df.train_dataset[df.train_dataset == 'snemi_200'] = 'snemi_25'
    df.average_precision = pd.to_numeric(df.average_precision)
    df.train_dataset = pd.Categorical(df.train_dataset)
    df.val_dataset = pd.Categorical(df.val_dataset)
    df.model = pd.Categorical(df.model)
    ds_size = np.array(
        [int(x.split('_')[-1]) for x in df.train_dataset.as_matrix()])
    df['ds'] = ds_size
    return df


def create_gen_plots(
        df,
        out_data,
        out_plot,
        order=['snemi_200', 'fib_200', 'berson_200']):
    """Plots for generalization analysis. cremi_200 is excluded."""
    palette = sns.cubehelix_palette(
        len(order), dark=0.4, light=0.8, reverse=True)
    import ipdb;ipdb.set_trace()
    # palette = sns.color_palette("Set2")  # 'colorblind'
    # f, ax = plt.subplots(
    #     1,
    #     1,
    #     sharey=True,
    #     figsize=(5, 5),
    #     gridspec_kw={'width_ratios': [1, 1]})
    sns.catplot(
        data=df,
        x='val_dataset',
        y='average_precision',
        hue='model',
        palette=palette,
        kind="bar",
        legend=False,
        # ax=ax,
        order=order)

    sns.set_style("white")
    sns.set_context("paper", font_scale=1)
    sns.despine(ax=ax[0], left=False, top=True, bottom=False, right=True)
    sns.barplot(
        data=df[df.model == 'seung_unet_per_pixel'],
        x='val_dataset',
        y='average_precision',
        palette=palette,
        ax=ax[0],
        kind='bar',
        order=order)
    ax[0].set_ylabel('Mean average precision')
    ax[0].set_title('UNet')
    ax[0].set_xlabel('')
    ax[0].set_xticks([0, 1, 2])
    ax[0].set_xticklabels(['SNEMI', 'FIB-25', 'Ding'])
    sns.despine(ax=ax[1], left=True, top=True, bottom=False, right=True)
    plt.ylim([0.75, 1])
    sns.barplot(
        data=df[df.model == 'gammanet_t8_per_pixel'],
        x='val_dataset',
        y='average_precision',
        palette=palette,
        order=order)
    ax[1].set_ylabel('')
    ax[1].tick_params(left=False)
    ax[1].set_title('$\gamma$Net')
    ax[1].set_xlabel('')
    ax[1].set_xticks([0, 1, 2])
    ax[1].set_xticklabels(['SNEMI', 'Ding', 'FIB-25'])
    plt.tight_layout()
    plt.savefig(os.path.join(out_data, out_plot))
    plt.show()
    plt.show()


def create_plots(
        df,
        out_data,
        out_plot='synth_sample_complexity.pdf',
        colors=None,
        hue_order=None,
        show_fig=False,
        ylim=(0.8, 1)):
    """Create figures for paper."""
    # Make sample complexity Fig 1
    fig, (ax, ax2) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(9, 7),
        gridspec_kw={'width_ratios': [5, 1]})
    sns.set_style("white")
    sns.set_context("paper", font_scale=1)

    # Ax1
    g = sns.lineplot(
        data=df[df.train_dataset != 'snemi_200'],
        x='ds',
        y='average_precision',
        hue='model',
        # legend='brief',
        ax=ax,
        palette=colors,
        hue_order=hue_order,
        # style='model',
        legend=False,
        # markers=['o'] * len(hue_order),
        dashes=False)  # , palette=sns.color_palette("mako_r", 4))
    (g.set(ylim=(0.8, 1)))
    ax.legend(labels=hue_order, frameon=False)
    ax.set_xticks([1, 5, 10, 20])
    ax.set_yticks([0.8, 0.9, 1.])
    ax.set_xticklabels(['1 (6.0)', '5 (6.7)', '10 (7)', '20 (7.3)'])
    ax.set_xlabel('Number of training images\n($log_{10}$ training samples)')
    ax.set_ylabel('Mean average precision')
    sns.despine()

    # Ax2
    new_df = df[df.train_dataset == 'snemi_200']
    ext_df = df[df.train_dataset == 'snemi_200']
    ext_df.train_dataset = 'snemi_190'
    ext_df.ds = 190
    new_df.train_dataset = 'snemi_210'
    new_df.ds = 210
    new_df = pd.concat((ext_df, new_df))
    sns.axes_style({'axes.spines.left': False})

    g2 = sns.lineplot(
        data=new_df,
        x='ds',
        y='average_precision',
        hue='model',
        legend=False,
        ax=ax2,
        palette=colors,
        hue_order=hue_order,
        # style='model',
        # markers=['o'] * len(hue_order),
        dashes=False)  # , palette=sns.color_palette("mako_r", 4))
    (g2.set(ylim=ylim))
    ax2.set_xticks([200])
    ax2.set_xticklabels(['200 (8.3)'])
    ax2.set_xlim([180, 220])
    ax2.set_xlabel('')
    g2.spines['left'].set_visible(False)
    # ax2.set_xlabel('Number of training examples')
    sns.despine()
    ax2.spines['left'].set_visible(False)
    g2.spines['left'].set_visible(False)
    ax2.tick_params(left=False)
    plt.tight_layout()

    # # Now add the break symbol in x axis
    # d = 0.010
    # kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    # ax.plot((1-d,1+d), (-d,+d), **kwargs)
    # kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
    # ax2.plot((-d,+d), (-d,+d), **kwargs)

    plt.savefig(os.path.join(out_data, out_plot))
    plt.show()
    plt.close(fig)


def plot_wrapper(
        data_files,
        keep_models,
        out_name,
        max_samples,
        out_data,
        generalization=False,
        create_subdir=True):
    """Wrapper for running routines."""
    print('Working on %s' % out_name)
    it_data_files = [x for x in data_files if out_name not in x]
    if create_subdir:
        py_utils.make_dir(os.path.join(out_data, out_name))
        out_name = '%s%s%s' % (out_name, os.path.sep, out_name)
    df, dm = gather_data(
        data_files=it_data_files,
        keep_models=keep_models,
        generalization=generalization)
    df = process_data(
        df=df,
        out_name=out_name,
        out_data=out_data,
        generalization=generalization,
        max_samples=max_samples)
    colors = keep_models.values()
    color_pal = sns.xkcd_palette(colors)
    hue_order = keep_models.keys()
    if generalization:
        create_gen_plots
        create_gen_plots(
            df=df,
            out_data=out_data,
            out_plot='%s.pdf' % out_name)
    else:
        create_plots(
            df=df,
            out_data=out_data,
            colors=color_pal,
            hue_order=hue_order,
            out_plot='%s.pdf' % out_name)
    print('Finished on %s' % out_name)


def main():
    """Create plots/csvs for the following experiments."""
    # Globals
    jk_data = 'maps'
    out_data = 'data_to_process_for_jk'
    data_files = glob(os.path.join(jk_data, '*.npz'))
    max_samples = 10

    # Main analysis: gamma-net versus seung models
    main_out_name = 'snemi_experiment_data'
    main_model_batch = {
        'gammanet_t8_per_pixel': 'scarlet',
        'seung_unet_per_pixel': 'dusty blue',
        'seung_unet_per_pixel_BSDS_init': 'deep sky blue'}
    # plot_wrapper(
    #     out_name=main_out_name,
    #     keep_models=main_model_batch,
    #     data_files=data_files,
    #     out_data=out_data,
    #     max_samples=max_samples)

    # Generalization analysis
    jk_data = 'test_maps'
    data_files = glob(os.path.join(jk_data, '*.npz'))
    main_out_name = 'generalize_snemi_experiment_data'
    main_model_batch = {
        'gammanet_t8_per_pixel': 'scarlet',
        'seung_unet_per_pixel': 'dusty blue'}
    plot_wrapper(
        out_name=main_out_name,
        keep_models=main_model_batch,
        data_files=data_files,
        out_data=out_data,
        max_samples=max_samples,
        generalization=True)


if __name__ == '__main__':
    main()

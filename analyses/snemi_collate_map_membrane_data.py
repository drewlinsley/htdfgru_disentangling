import os
import numpy as np
import pandas as pd
from glob import glob
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns
from utils import py_utils
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels import stats
from scipy.stats import ttest_ind


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
            if ('snemi' in config.train_dataset) or \
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
                try:
                    maps = d['maps']
                    # arands = d['arands']
                    map_len = 1  # len(maps)
                    dm += [np.vstack([
                        # [idx] * map_len,
                        np.mean(maps),
                        # np.mean(arands),
                        [config.train_dataset] * map_len,
                        [config.val_dataset] * map_len,
                        [f] * map_len,
                        [ckpt] * map_len,
                        [config.model] * map_len
                    ])]
                except Exception:
                    print 'Failed to load %s' % f
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
    if generalization:
        columns = [
            # 'model_idx',
            'average_precision',
            # 'arand',
            'train_dataset',
            'val_dataset',
            'file_name',
            'ckpt',
            'model']
    else:
        columns = [
            # 'model_idx',
            'average_precision',
            'train_dataset',
            'val_dataset',
            'file_name',
            'ckpt',
            'model']
    df = pd.DataFrame(
        np.concatenate(dm, axis=1).transpose(),
        columns=columns)
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
                if 'snemi' in va:
                    it_df = it_df.sort_values(
                        'average_precision', ascending=False)
                    it_mask = np.ones(len(it_df.file_name), dtype=bool)
                    it_mask[max_samples:] = False
                    proc_df += [it_df[it_mask]]
                else:
                    it_mask = np.ones(len(it_df.file_name), dtype=bool)
                    # it_mask[it_df.ckpt == 'False'] = False
                    it_mask[max_samples:] = False
                    proc_df += [it_df[it_mask]]

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
                    proc_df += [it_df[it_mask]]
    df = pd.concat(proc_df)
    df[df.train_dataset == 'snemi_combos_200'] = 'snemi_200'
    import ipdb;ipdb.set_trace()  # df[np.logical_and(df.ckpt != 'False', df.train_dataset == 'snemi_200')]
    if generalization:
        df.train_dataset[df.train_dataset == 'snemi_200_test'] = 'snemi_200'
        df.val_dataset[df.val_dataset == 'snemi_200_test'] = 'snemi_200'
    # df.model[df.model == 'seung_unet_per_pixel'] = 'UNet'
    # df.model[df.model == 'gammanet_t8_per_pixel'] = r'$\gamma$Net'

    # Recode the datasets
    # df.train_dataset[df.train_dataset == 'snemi_200'] = 'snemi_25'
    df.average_precision = pd.to_numeric(df.average_precision)
    df.train_dataset = pd.Categorical(df.train_dataset)
    df.val_dataset = pd.Categorical(df.val_dataset)
    df.model = pd.Categorical(df.model)
    ds_size = np.array(
        [int(x.split('_')[-1]) for x in df.train_dataset.as_matrix()])
    df['ds'] = ds_size

    # Group data
    if generalization:
        grouped_data = df.groupby(
            ['val_dataset', 'model']).agg('max').reset_index()
    else:
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
        max_selection = ckpts_data.groupby(
            ['train_dataset', 'val_dataset', 'model']).agg(
                'max').reset_index()
        ckpts_data.to_csv(os.path.join(out_data, '%s_ckpts.csv' % out_name))
        ckpts_data['model'].to_csv(
            os.path.join(out_data, '%s_models_only.csv' % out_name))
        ckpts_data['ckpt'].to_csv(
            os.path.join(out_data, '%s_ckpts_only.csv' % out_name))
        ckpts_data['train_dataset'].to_csv(
            os.path.join(out_data, '%s_trainds_only.csv' % out_name))
        max_selection.to_csv(
            os.path.join(out_data, '%s_max_ckpts.csv' % out_name))
        max_selection['model'].to_csv(
            os.path.join(out_data, '%s_max_models_only.csv' % out_name))
        max_selection['ckpt'].to_csv(
            os.path.join(out_data, '%s_max_ckpts_only.csv' % out_name))
        max_selection['train_dataset'].to_csv(
            os.path.join(out_data, '%s_max_trainds_only.csv' % out_name))
    return df


def create_gen_plots(
        df,
        out_data,
        out_plot,
        order=['snemi_200', 'fib_200', 'berson_200']):
    """Plots for generalization analysis. cremi_200 is excluded."""
    # Run an anova + pairwise comps
    mod = ols(
        'average_precision ~ C(model) + C(val_dataset) + C(model):C(val_dataset)',
        data=df).fit()
    aov_table = sm.stats.anova_lm(mod, typ=2)
    print(aov_table)
    ds_stats = stats.multicomp.MultiComparison(
        df['average_precision'],
        df['val_dataset'])
    ds_rtp = ds_stats.allpairtest(ttest_ind, method='b')[0]
    print(ds_rtp)
    all_mc_ts = []
    for ds in order:
        mc = stats.multicomp.MultiComparison(
            df['average_precision'][df.val_dataset == 'snemi_200'],
            df['model'][df.val_dataset == 'snemi_200'])
        mc = mc.allpairtest(ttest_ind, method='b')[0]
        print(ds)
        print(mc)
        all_mc_ts += [mc]
    print(all_mc_ts)

    # Now plot everything
    # palette = sns.cubehelix_palette(
    #     len(order), dark=0.1, light=0.95, reverse=True)
    palette = sns.xkcd_palette(['scarlet', 'dusty blue'])
    sns.set_style("ticks")  # "white")
    sns.set_context("paper", font_scale=1)
    f, ax = plt.subplots(1, 1, tight_layout=True, figsize=(8, 4))
    sns.barplot(
        data=df,
        x='val_dataset',
        y='average_precision',
        hue='model',
        palette=palette,
        # kind="bar",
        # legend=False,
        # errcolor='0.35',
        # legend_out=False,
        ax=ax,
        order=order)
    ax.set_ylim([0.7, 1])
    ax.set_ylabel('Mean average precision')
    ax.set_xlabel('')
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0.7, 0.8, 0.9, 1.])
    ax.set_xticklabels(['SNEMI', 'FIB-25', 'Ding'])
    sns.despine(ax=ax)
    # plt.legend(labels=['UNet', ''])
    # plt.tight_layout()
    plt.savefig(os.path.join(out_data, out_plot))
    plt.show()


def create_plots(
        df,
        out_data,
        out_plot='synth_sample_complexity.pdf',
        colors=None,
        hue_order=None,
        show_fig=False,
        ylim=(0.5, 1)):  # (0.3, 1)):  # (0.8, 1)
    """Create figures for paper."""
    # Make sample complexity Fig 1
    fig, (ax, ax2) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(5, 5),
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
    (g.set(
        ylim=ylim,
        # yticks=[0.8, 0.85, 0.9, 0.95, 1.],
        # yticklabels=[0.8, 0.85, 0.9, 0.95, 1.],
        xticks=[1, 5, 10, 20],
        xticklabels=['1 (6.0)', '5 (6.7)', '10 (7)', '20 (7.3)'],
        xlabel='Number of training images\n($log_{10}$ training samples)',
        ylabel='Mean average precision'))
    ax.legend(labels=hue_order, frameon=False)
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
    (g2.set(
        ylim=ylim,
        xticks=[200],
        xticklabels=['200 (8.3)'],
        xlim=[180, 220],
        xlabel='',
        ylabel=''))
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
    max_samples = 9

    # Generalization analysis
    data_files = glob(os.path.join('test_maps', '*.npz'))  #  + glob(os.path.join(jk_data, '*.npz'))
    data_files.sort(key=os.path.getmtime)
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

    # Main analysis: gamma-net versus seung models
    data_files = glob(os.path.join(jk_data, '*.npz'))
    main_out_name = 'snemi_experiment_data'
    main_model_batch = {
        'gammanet_t8_per_pixel': 'scarlet',
        'seung_unet_per_pixel': 'dusty blue'}
    plot_wrapper(
        out_name=main_out_name,
        keep_models=main_model_batch,
        data_files=data_files,
        out_data=out_data,
        max_samples=max_samples)

    # Main analysis: gamma-net versus seung models
    main_out_name = 'snemi_experiment_data_with_BSDS'
    main_model_batch = {
        'gammanet_t8_per_pixel': 'scarlet',
        'seung_unet_per_pixel': 'dusty blue',
        'seung_unet_per_pixel_BSDS_init': 'grey'}
    plot_wrapper(
        out_name=main_out_name,
        keep_models=main_model_batch,
        data_files=data_files,
        out_data=out_data,
        max_samples=max_samples)


if __name__ == '__main__':
    main()

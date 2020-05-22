import os
import numpy as np
import pandas as pd
from glob import glob
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns
from utils import py_utils


def plot_invariance_idx(proc_data, out_data, out_name, y='scale diff'):
    """Plot invariance index lineplots."""
    f, ax = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(7, 2.5),
        gridspec_kw={'width_ratios': [1, 1]})
    sns.set_style("white")
    sns.set_context("paper", font_scale=1, rc={"lines.linewidth": 2.5})
    sns.despine(top=True, right=True)
    # sns.despine(ax=ax[0], left=False, top=False, bottom=True, right=True)
    g = sns.lineplot(
        data=proc_data[proc_data.model == 'unet'],
        hue='train set',
        x='variation',
        y=y,
        style='train set',
        legend=False,  # 'brief',
        palette=sns.color_palette("GnBu", len(np.unique(proc_data['train set']))),  # palette,  # sns.light_palette("blue", 5, reverse=True),
        ax=ax[0])
    # (g.set(ylim=(-0.8, 0.1, )))
    (g.set(ylim=(-0.1, 0.8)))
    ax[0].set_ylabel('')
    # ax[0].set_xlabel('')
    ax[0].set_xlabel('UNet')
    ax[0].set_xticks([1, 2, 3])
    ax[0].set_xticklabels([r'$\times$1', r'$\times$2', r'$\times$3'])
    ax[0].set_yticklabels(['{:,.0%}'.format(x) for x in ax[0].get_yticks()])
    # Put the legend out of the figure
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    g = sns.lineplot(
        data=proc_data[proc_data.model == 'gamma'],
        hue='train set',
        x='variation',
        y=y,
        style='train set',
        legend=False,  # 'brief',
        palette=sns.color_palette("OrRd", len(np.unique(proc_data['train set']))),  # sns.light_palette("red", 5, reverse=True),
        ax=ax[1])
    # sns.despine(ax=ax[1], left=True, top=False, bottom=True, right=True)
    sns.despine(ax=ax[1], left=True, top=True, right=True)
    # (g.set(ylim=(-0.8, 0.1)))
    (g.set(ylim=(-0.1, 0.8)))
    ax[1].set_ylabel('')
    ax[1].tick_params(left=False)
    # ax[1].set_xlabel('')
    ax[1].set_xlabel(r'$\gamma$Net')
    ax[1].set_xticks([1, 2, 3])
    ax[1].set_xticklabels([r'$\times$1', r'$\times$2', r'$\times$3'])
    f.text(
        0.05,
        0.5,
        'Performance difference from baseline',
        ha='center',
        va='center',
        rotation='vertical')
    f.text(
        0.55,
        0.96,
        'Cell %s from baseline' % y,
        ha='center',
        va='center')
    # Put the legend out of the figure
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(
        os.path.join(
            out_data, '%s_deviation_%s.pdf' % (out_name, '_'.join(y))))
    plt.show()
    plt.close(f)

    # Plot X v Y global
    scatter_df = pd.DataFrame(np.hstack((proc_data[proc_data.model == 'unet']['global diff'].values.reshape(-1, 1), proc_data[proc_data.model == 'gamma']['global diff'].values.reshape(-1, 1), proc_data[proc_data.model == 'unet']['variation'].values.reshape(-1, 1), proc_data[proc_data.model == 'unet']['train set'].values.reshape(-1, 1))), columns=['unet', 'gamma', 'variation', 'train'])
    g = sns.FacetGrid(scatter_df, col='variation', hue='train').map(plt.scatter, 'unet', 'gamma').set(xlim=(-0.05, 1), ylim=(-0.05, 1))
    plt.title('Gamma')
    plt.savefig(
        os.path.join(
            out_data, '%s_deviation_global_%s.pdf' % (out_name, '_'.join(y))))
    plt.show()

    # Plot X v Y scale
    scatter_df = pd.DataFrame(np.hstack((proc_data[proc_data.model == 'unet']['scale diff'].values.reshape(-1, 1), proc_data[proc_data.model == 'gamma']['scale diff'].values.reshape(-1, 1), proc_data[proc_data.model == 'unet']['variation'].values.reshape(-1, 1), proc_data[proc_data.model == 'unet']['train set'].values.reshape(-1, 1))), columns=['unet', 'gamma', 'variation', 'train'])
    g = sns.FacetGrid(scatter_df, col='variation', hue='train').map(plt.scatter, 'unet', 'gamma').set(xlim=(-0.05, 1), ylim=(-0.05, 1))
    plt.title('Gamma')
    plt.savefig(
        os.path.join(
            out_data, '%s_deviation_scale_%s.pdf' % (out_name, '_'.join(y))))
    plt.show()


def create_plots(
        df,
        out_data,
        out_plot='synth_sample_complexity.pdf',
        colors=None,
        hue_order=None,
        show_fig=False):
    """Create figures for paper."""
    # Make sample complexity Fig 1
    fig, (ax, ax2) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(9, 5),
        gridspec_kw={'width_ratios': [5, 1]})
    sns.set_style("white")
    sns.set_context("paper", font_scale=1)

    # Ax1
    g = sns.lineplot(
        data=df[np.logical_and(
            df.train_dataset != 'Baseline, N=200', df.val_dataset == 'Same')],
        x='ds',
        y='average_precision',
        hue='model',
        # legend='brief',
        ax=ax,
        palette=colors,
        hue_order=hue_order,
        legend=False,
        markers=True,
        dashes=False)  # , palette=sns.color_palette("mako_r", 4))
    (g.set(ylim=(0.25, 1)))
    ax.legend(labels=hue_order, frameon=False)
    ax.set_xticks([1, 5, 10, 20])
    ax.set_xticklabels(['1 (5.0)', '5 (5.7)', '10 (6.0)', '20 (6.3)'])
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
    ax.set_xlabel('Number of training images\n($log_{10}$ training samples)')
    ax.set_ylabel('Mean average precision')
    sns.despine()

    # Ax2
    new_df = df[df.train_dataset == 'Baseline, N=200']
    ext_df = df[df.train_dataset == 'Baseline, N=200']
    ext_df.train_dataset = 'Baseline, N=190'
    ext_df.ds = 190
    new_df.train_dataset = 'Baseline, N=210'
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
        markers=True,
        dashes=False)  # , palette=sns.color_palette("mako_r", 4))
    # (g2.set(ylim=(0.8, 1)))
    ax2.set_xticks([200])
    ax2.set_xticklabels(['200 (7.3)'])
    ax2.set_xlim([180, 220])
    ax2.set_xlabel('')
    g2.spines['left'].set_visible(False)
    # ax2.set_xlabel('Number of training examples')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(out_data, out_plot))
    if show_fig:
        plt.show()
    plt.close(fig)


def gather_data(data_files, keep_models):
    """Gather data from npzs in data_files."""
    # Start data proc loop
    # val_images, val_labels, val_logits = [], [], []
    # model_names, train_datasets = [], []
    dm = []
    keep_models = keep_models.keys()
    for idx, f in tqdm(
            enumerate(data_files),
            total=len(data_files),
            desc='Gathering data for JK'):
        try:
            d = np.load(f)
            config = d['config'].item()
        except Exception as e:
            print 'Failed on %s with %s' % (f, e)
        if check_model(
                config.model, keep_models) or \
                'snemi' in config.train_dataset or \
                'cremi' in config.train_dataset or \
                'fib' in config.train_dataset or \
                'berson' in config.train_dataset:
            continue
        else:
            # model_names += [config.model]
            # train_datasets += [config.train_dataset]
            srch = glob(
                os.path.join(
                    '/media/data_cifs/cluttered_nist_experiments/checkpoints',
                    '%s' % f.split(os.path.sep)[-1].split('.')[0],
                    '*.ckpt*.meta'))
            if len(srch):
                ckpt = srch[0].split('.meta')[0]
            else:
                ckpt = False

            # STORE THE OG TRAIN SET HERE
            try:
                maps = d['maps']
                arands = d['arands']
                map_len = 1  # len(maps)
                dm += [np.vstack([
                    # [idx] * map_len,
                    np.mean(maps),
                    np.mean(arands),
                    [config.train_dataset] * map_len,
                    [config.val_dataset] * map_len,
                    [f] * map_len,
                    [ckpt] * map_len,
                    [config.extra_0] * map_len,
                    [config.model] * map_len
                ])]
            except Exception as e:
                print('Failed on %s: %s' % (f, e))
    df = pd.DataFrame(
        np.concatenate(dm, axis=1).transpose(),
        columns=[
            # 'model_idx',
            'average_precision',
            'arand',
            'train_dataset',
            'val_dataset',
            'file_name',
            'ckpt',
            'original_train',
            'model'])
    return df, dm


def check_model(model_name, model_list):
    """Return False if model_name in model_list."""
    for m in model_list:
        if model_name == m:
            return False
    return True


def process_data(df, out_name, out_data, baseline_file, max_samples=10, threshold=0.55):
    """Prepare data for saving to csv + plotting."""
    # Process df so that no model has > max_samples entries
    unique_train = np.unique(df.train_dataset)
    unique_val = np.unique(df.val_dataset)
    unique_model = np.unique(df.model)
    proc_df = []
    for tr in unique_train:
        for va in unique_val:
            for mo in unique_model:
                idx = np.logical_and(
                    np.logical_and(
                        df.train_dataset == tr, df.val_dataset == va),
                    df.model == mo)
                it_df = df[idx]
                # it_df['average_precision'] = it_df.average_precision.fillna(value=-1)
                it_df = it_df.sort_values('average_precision', ascending=False)
                it_mask = np.ones(len(it_df.file_name), dtype=bool)
                it_mask[max_samples:] = False
                proc_df += [it_df]  # [it_mask]]
    df = pd.concat(proc_df)

    # Group data
    grouped_data = df.groupby(
        ['train_dataset', 'val_dataset', 'model']).agg('max').reset_index()

    # Save csvs
    df.to_csv(os.path.join(out_data, '%s_raw.csv' % out_name))
    grouped_data.to_csv(os.path.join(out_data, '%s_grouped.csv' % out_name))

    # Recode the datasets
    df.average_precision = pd.to_numeric(df.average_precision)
    df.arand = pd.to_numeric(df.arand)
    df.val_dataset[
        df.val_dataset == 'v2_synth_connectomics_size_1'] = 'Scale +1'
    df.val_dataset[
        df.val_dataset == 'v2_synth_connectomics_size_2'] = 'Scale +2'
    df.val_dataset[
        df.val_dataset == 'v2_synth_connectomics_size_3'] = 'Scale +3'
    df.val_dataset[
        df.val_dataset == 'v2_synth_connectomics_lumcontrast_1'] = 'Global +1'
    df.val_dataset[
        df.val_dataset == 'v2_synth_connectomics_lumcontrast_2'] = 'Global +2'
    df.val_dataset[
        df.val_dataset == 'v2_synth_connectomics_lumcontrast_3'] = 'Global +3'
    df.val_dataset[
        df.val_dataset == 'v2_synth_connectomics_baseline_20_test'] = 'Baseline +0'
    df.train_dataset[
        df.train_dataset == 'v2_synth_connectomics_baseline_20_test'] = 'Baseline_20'
    df.train_dataset = pd.Categorical(df.train_dataset)
    df.val_dataset = pd.Categorical(df.val_dataset)
    df.model = pd.Categorical(df.model)

    ds_size = np.array(
        [int(x.split('_')[-1]) for x in df.train_dataset.as_matrix()])
    df['ds'] = ds_size

    # Get val-dataset diffs for parameterized straining
    # unique_models = np.unique(df.model)
    unique_ds = np.unique(df.ds)
    invs = ['Scale', 'Global']
    fix_unique_ds = []
    for ds in unique_ds:
        for inv in invs:
            fix_unique_ds += ['%s +%s' % (inv, ds)]

    # Second stage recoding
    df.original_train[
        df.original_train == 'v2_synth_connectomics_baseline_1'] = 1
    df.original_train[
        df.original_train == 'v2_synth_connectomics_baseline_5'] = 5
    df.original_train[
        df.original_train == 'v2_synth_connectomics_baseline_10'] = 10
    df.original_train[
        df.original_train == 'v2_synth_connectomics_baseline_20'] = 20
    df.original_train[
        df.original_train == 'v2_synth_connectomics_baseline_200'] = 200
    df.original_train = pd.to_numeric(df.original_train)
    # # # df = df[df.original_train < 200]

    agg_gamma = df[df.model == 'gammanet_t8_per_pixel'].groupby(
        ['val_dataset', 'original_train'])['average_precision'].agg(
        np.nanmax).reset_index()
    agg_unet = df[df.model == 'seung_unet_per_pixel'].groupby(
        ['val_dataset', 'original_train'])['average_precision'].agg(
        np.nanmax).reset_index()
    arand_gamma = df[df.model == 'gammanet_t8_per_pixel'].groupby(
        ['val_dataset', 'original_train'])['arand'].agg(
        np.nanmax).reset_index()
    arand_unet = df[df.model == 'seung_unet_per_pixel'].groupby(
        ['val_dataset', 'original_train'])['arand'].agg(
        np.nanmax).reset_index()

    diff_perf = ''  # 'arand'
    if diff_perf == 'arand':
        diff_gamma = arand_gamma
        diff_unet = arand_unet
        diff_gamma['average_precision'] = diff_gamma.arand
        diff_unet['average_precision'] = diff_unet.arand
    else:
        diff_gamma = agg_gamma
        diff_unet = agg_unet

    trains = np.unique(df.original_train)
    proc_data = []
    for tr in trains:
        it_gamma = diff_gamma[diff_gamma.original_train == tr]
        it_unet = diff_unet[diff_unet.original_train == tr]
        # gamma_baseline = it_gamma[
        #     it_gamma.val_dataset == 'Baseline +0'].average_precision.values[0]
        # unet_baseline = it_unet[
        #     it_unet.val_dataset == 'Baseline +0'].average_precision.values[0]
        gamma_baseline = diff_gamma[
            diff_gamma.val_dataset == 'Baseline +0'].average_precision.max()
        unet_baseline = diff_unet[
            diff_unet.val_dataset == 'Baseline +0'].average_precision.max()
        for idx in range(1, 4):
            gamma_global = it_gamma[
                it_gamma.val_dataset == 'Global +%s' % idx].average_precision.mean()
            gamma_scale = it_gamma[
                it_gamma.val_dataset == 'Scale +%s' % idx].average_precision.mean()
            unet_global = it_unet[
                it_unet.val_dataset == 'Global +%s' % idx].average_precision.mean()
            unet_scale = it_unet[
                it_unet.val_dataset == 'Scale +%s' % idx].average_precision.mean()
            unet_baseline = np.maximum(unet_baseline, gamma_baseline)
            gamma_baseline = unet_baseline
            gamma_global_diff = gamma_baseline - gamma_global
            gamma_global_norm = gamma_baseline / (
                gamma_baseline - gamma_global)
            gamma_scale_diff = gamma_baseline - gamma_scale
            gamma_scale_norm = gamma_baseline / (
                gamma_baseline - gamma_scale)
            unet_global_diff = unet_baseline - unet_global
            unet_global_norm = unet_baseline / (
                unet_baseline - unet_global)
            unet_scale_diff = unet_baseline - unet_scale
            unet_scale_norm = unet_baseline / (
                unet_baseline - unet_scale)
            proc_data += [[
                tr,
                idx,
                'gamma',
                gamma_global_diff,
                gamma_scale_diff,
                gamma_global_norm,
                gamma_scale_norm
            ]]
            proc_data += [[
                tr,
                idx,
                'unet',
                unet_global_diff,
                unet_scale_diff,
                unet_global_norm,
                unet_scale_norm
            ]]
    proc_data = pd.DataFrame(
        proc_data,
        columns=[
            'train set',
            'variation',
            'model',
            'global diff',
            'scale diff',
            'global_norm',
            'scale_norm'])
    # proc_data['train set'][proc_data['train set'] == 5] = 2
    # proc_data['train set'][proc_data['train set'] == 10] = 3
    # proc_data['train set'][proc_data['train set'] == 20] = 4
    # proc_data['train set'][proc_data['train set'] == 200] = 5
    plot_invariance_idx(proc_data, out_data, out_name, y='scale diff')
    plot_invariance_idx(proc_data, out_data, out_name, y='global diff')

    os._exit(1)

    # Count # > threshold
    thresh_df = df.copy()
    thresh_df.average_precision = (thresh_df.average_precision > threshold).astype(int)
    thresh_gamma = thresh_df[thresh_df.model == 'gammanet_t8_per_pixel'].groupby(
        ['val_dataset', 'original_train'])['average_precision'].agg(
        np.nanmean).reset_index()
    thresh_unet = thresh_df[thresh_df.model == 'seung_unet_per_pixel'].groupby(
        ['val_dataset', 'original_train'])['average_precision'].agg(
        np.nanmean).reset_index()
    plt.subplot(121)
    plt.title('UNet')
    sns.lineplot(
        data=thresh_unet,
        hue='val_dataset',
        y='average_precision',
        x='original_train')
    plt.ylim([-0.1, 1.1])
    plt.subplot(122)
    plt.title('Gammanet')
    sns.lineplot(
        data=thresh_gamma,
        hue='val_dataset',
        y='average_precision',
        x='original_train')
    plt.ylim([0, 1])
    plt.show()
    plt.close('all')

    baseline_file = baseline_file.groupby(
        ['train_dataset', 'model'])['average_precision'].max().reset_index()
    baseline_file.train_dataset[
        baseline_file.train_dataset == 'v2_synth_connectomics_baseline_1'] = 1
    baseline_file.train_dataset[
        baseline_file.train_dataset == 'v2_synth_connectomics_baseline_5'] = 5
    baseline_file.train_dataset[
        baseline_file.train_dataset == 'v2_synth_connectomics_baseline_10'] = 10
    baseline_file.train_dataset[
        baseline_file.train_dataset == 'v2_synth_connectomics_baseline_20'] = 20
    baseline_file.train_dataset[
        baseline_file.train_dataset == 'v2_synth_connectomics_baseline_200'] = 200
    baseline_file.train_dataset = pd.to_numeric(baseline_file.train_dataset)
    gamma_diffs, unet_diffs = [], []
    unique_train = np.unique(baseline_file.train_dataset)
    for tr in unique_train:
        it_baseline = baseline_file[baseline_file.train_dataset == tr]
        gamma_score = agg_gamma[
            agg_gamma.original_train == tr].average_precision.as_matrix()
        unet_score = agg_unet[
            agg_unet.original_train == tr].average_precision.as_matrix()
        baseline_unet = it_baseline[
            it_baseline.model == 'seung_unet_per_pixel'].average_precision.as_matrix()
        baseline_gamma = it_baseline[
            it_baseline.model == 'gammanet_t8_per_pixel'].average_precision.as_matrix()
        it_gamma = baseline_gamma - gamma_score
        it_unet = baseline_unet - unet_score
        stack_gamma = agg_gamma[agg_gamma.original_train == tr].copy()
        stack_unet = agg_unet[agg_unet.original_train == tr].copy()
        stack_gamma.average_precision = it_gamma
        stack_unet.average_precision = it_unet
        gamma_diffs += [stack_gamma]
        unet_diffs += [stack_unet]
    gamma_diffs = pd.concat(gamma_diffs)
    unet_diffs = pd.concat(unet_diffs)
    plt.subplot(121);sns.lineplot(data=gamma_diffs, x='original_train', hue='val_dataset', y='average_precision');plt.ylim([-0.1, 0.5]);plt.subplot(122);sns.lineplot(data=unet_diffs, x='original_train', hue='val_dataset', y='average_precision');plt.ylim([-0.1, 0.5]);plt.show() 

    # sns.lineplot(
    # data=diff_df, x='gv_diff', y='sv_diff', hue='model');plt.show()
    return df, diff_df


def plot_wrapper(
        data_files,
        keep_models,
        out_name,
        max_samples,
        baseline_file,
        out_data,
        create_subdir=True):
    """Wrapper for running routines."""
    print('Working on %s' % out_name)
    it_data_files = [x for x in data_files if out_name not in x]
    if create_subdir:
        py_utils.make_dir(os.path.join(out_data, out_name))
        out_name = '%s%s%s' % (out_name, os.path.sep, out_name)
    df, dm = gather_data(data_files=it_data_files, keep_models=keep_models)
    baseline_file = pd.read_csv(baseline_file)
    df, diff_df = process_data(
        df=df,
        out_name=out_name,
        out_data=out_data,
        baseline_file=baseline_file,
        max_samples=max_samples)

    # Create color maps
    colors = keep_models.values()
    color_pal = sns.xkcd_palette(colors)
    hue_order = keep_models.keys()
    create_plots(df=df, out_data=out_data, colors=color_pal, hue_order=hue_order, out_plot='%s.pdf' % out_name)
    print('Finished on %s' % out_name)


def main():
    """Create plots/csvs for the following experiments."""

    # Globals
    jk_data = 'test_maps'
    out_data = 'data_to_process_for_jk'
    data_files = glob(os.path.join(jk_data, '*.npz'))
    baseline_file = os.path.join(out_data, 'inv_main_synth_experiment_data/inv_main_synth_experiment_data_ckpts.csv')
    max_samples = 10

    # Main analysis: gamma-net versus seung models
    main_out_name = 'inv_main_synth_experiment_data'
    main_model_batch = {
        'gammanet_t8_per_pixel': 'scarlet',
        'seung_unet_per_pixel': 'dusty blue'}  # ,
        # 'ffn_per_pixel': 'kelly green'}
    plot_wrapper(
        out_name=main_out_name,
        keep_models=main_model_batch,
        data_files=glob(os.path.join(jk_data, '*.npz')),
        out_data=out_data,
        baseline_file=baseline_file,
        max_samples=max_samples)


if __name__ == '__main__':
    main()


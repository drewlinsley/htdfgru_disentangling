import os
import numpy as np
import pandas as pd
from glob import glob
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns
from utils import py_utils


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
        # figsize=(10, 7),
        figsize=(5, 5),
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
    # (g.set(ylim=(0.25, 1)))
    (g.set(ylim=(0.5, 1)))
    ax.legend(labels=hue_order, frameon=False)
    ax.set_xticks([1, 5, 10, 20])
    ax.set_xticklabels(['1 (5.0)', '5 (5.7)', '10 (6.0)', '20 (6.3)'])
    # ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.])
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
    sns.despine(ax=ax2, left=True)
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
                config.model, keep_models) or 'snemi' in config.train_dataset:
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
            try:
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
            except Exception as e:
                print('Failed on %s: %s' % (f, e))
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


def process_data(df, out_name, out_data, max_samples=10):
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
                it_df = it_df.sort_values('average_precision', ascending=False)
                it_mask = np.ones(len(it_df.file_name), dtype=bool)
                it_mask[max_samples:] = False
                proc_df += [it_df[it_mask]]
    df = pd.concat(proc_df)

    # Group data
    grouped_data = df.groupby(
        ['train_dataset', 'val_dataset', 'model']).agg('max').reset_index()

    # Ckpt data
    ckpts_data = df[np.logical_and(df.ckpt.str.contains('/media'), df.val_dataset.str.contains('baseline'))].groupby(
        ['train_dataset', 'val_dataset', 'model', 'ckpt']).agg(
        'max').reset_index()

    # Select model/ckpts for invariance analysis
    # selection = np.logical_and(np.logical_and(df.ckpt.str.contains('/media'), df.val_dataset.str.contains('baseline')), np.logical_or(np.logical_or(df.model == 'gammanet_t8_per_pixel', df.model == 'seung_unet_per_pixel'), df.model == 'ffn_per_pixel'))
    selection = np.logical_and(df.ckpt.str.contains('/media'), df.val_dataset.str.contains('baseline'))
    select_data = df[selection]
    max_selection = select_data.groupby(['train_dataset', 'val_dataset', 'model']).agg(
        'max').reset_index()

    # Save csvs
    df.to_csv(os.path.join(out_data, '%s_raw.csv' % out_name))
    grouped_data.to_csv(os.path.join(out_data, '%s_grouped.csv' % out_name))
    select_data.to_csv(os.path.join(out_data, '%s_ckpts.csv' % out_name))
    select_data['model'].to_csv(os.path.join(out_data, '%s_models_only.csv' % out_name))
    select_data['ckpt'].to_csv(os.path.join(out_data, '%s_ckpts_only.csv' % out_name))
    select_data['train_dataset'].to_csv(os.path.join(out_data, '%s_trainds_only.csv' % out_name))
    max_selection.to_csv(os.path.join(out_data, '%s_max_ckpts.csv' % out_name))
    max_selection['model'].to_csv(os.path.join(out_data, '%s_max_models_only.csv' % out_name))
    max_selection['ckpt'].to_csv(os.path.join(out_data, '%s_max_ckpts_only.csv' % out_name))
    max_selection['train_dataset'].to_csv(os.path.join(out_data, '%s_max_trainds_only.csv' % out_name))

   # Recode the datasets
    df.average_precision = pd.to_numeric(df.average_precision)
    df.train_dataset[
        df.train_dataset == 'v2_synth_connectomics_baseline_1'] = 'Baseline, N=1'
    df.train_dataset[
        df.train_dataset == 'v2_synth_connectomics_baseline_5'] = 'Baseline, N=5'
    df.train_dataset[
        df.train_dataset == 'v2_synth_connectomics_baseline_10'] = 'Baseline, N=10'
    df.train_dataset[
        df.train_dataset == 'v2_synth_connectomics_baseline_20'] = 'Baseline, N=20'
    df.train_dataset[
        df.train_dataset == 'v2_synth_connectomics_baseline_200'] = 'Baseline, N=200'
    df.val_dataset[
        df.val_dataset == 'v2_synth_connectomics_baseline_20'] = 'Same'
    df.val_dataset[
        df.val_dataset == 'v2_synth_connectomics_lumcontrast'] = 'Global variation'
    df.val_dataset[
        df.val_dataset == 'v2_synth_connectomics_size'] = 'Size variation'
    df.train_dataset = pd.Categorical(df.train_dataset)
    df.val_dataset = pd.Categorical(df.val_dataset)
    df.model = pd.Categorical(df.model)
    ds_size = np.array(
        [int(x.split('=')[-1]) for x in df.train_dataset.as_matrix()])
    df['ds'] = ds_size

    # Get val-dataset diffs for parameterized straining
    unique_models = np.unique(df.model)
    unique_ds = np.unique(df.ds)
    diffs = []
    log_transform = False
    for model in unique_models:
        for ds in unique_ds:
            it_df = df[np.logical_and(df.model == model, df.ds == ds)]
            gv = it_df[it_df.val_dataset == 'Global variation']
            sv = it_df[it_df.val_dataset == 'Size variation']
            bv = it_df[it_df.val_dataset == 'Same']
            if log_transform:
                gv.average_precision = np.log10(gv.average_precision + 1)
                sv.average_precision = np.log10(sv.average_precision + 1)
                bv.average_precision = np.log10(bv.average_precision + 1)
            gv_ap = gv.average_precision
            sv_ap = sv.average_precision
            bv_ap = bv.average_precision
            gv_diff = (gv_ap.mean() - bv_ap.mean())
            sv_diff = (sv_ap.mean() - bv_ap.mean())
            diffs += [[model, ds, gv_diff, sv_diff]]
    diff_df = pd.DataFrame(
        diffs, columns=['model', 'ds', 'gv_diff', 'sv_diff'])
    # sns.lineplot(
    # data=diff_df, x='gv_diff', y='sv_diff', hue='model');plt.show()
    return df, diff_df


def plot_wrapper(
        data_files,
        keep_models,
        out_name,
        max_samples,
        out_data,
        create_subdir=True):
    """Wrapper for running routines."""
    print('Working on %s' % out_name)
    it_data_files = [x for x in data_files if out_name not in x]
    if create_subdir:
        py_utils.make_dir(os.path.join(out_data, out_name))
        out_name = '%s%s%s' % (out_name, os.path.sep, out_name)
    df, dm = gather_data(data_files=it_data_files, keep_models=keep_models)
    df, diff_df = process_data(
        df=df,
        out_name=out_name,
        out_data=out_data,
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
    jk_data = 'maps'
    out_data = 'data_to_process_for_jk'
    data_files = glob(os.path.join(jk_data, '*.npz'))
    max_samples = 10

    # Teaser fig: gamma-nets versus seung models
    main_out_name = 'teaser_main_synth_experiment_data'
    main_model_batch = {
        'gammanet_t8_per_pixel': 'scarlet',
        'gammanet_t4_per_pixel': 'red orange',
        'gammanet_t1_per_pixel': 'orange',
        'seung_unet_per_pixel': 'dusty blue'}  # ,
        # 'ffn_per_pixel': 'kelly green'}
    plot_wrapper(
        out_name=main_out_name,
        keep_models=main_model_batch,
        data_files=glob(os.path.join(jk_data, '*.npz')),
        out_data=out_data,
        max_samples=max_samples)

    # Invariance analysis: gamma-net versus seung models
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
        max_samples=max_samples)

    # Main analysis: gamma-net versus seung models
    main_out_name = 'main_synth_experiment_data'
    main_model_batch = {
        'gammanet_t8_per_pixel': 'scarlet',
        'seung_unet_per_pixel': 'dusty blue',
        'seung_unet_per_pixel_BSDS_init': 'deep sky blue'}
        # 'ffn_per_pixel': 'kelly green',
        # 'ffn_per_pixel_BSDS_init': 'darkgreen'}
    plot_wrapper(
        out_name=main_out_name,
        keep_models=main_model_batch,
        data_files=glob(os.path.join(jk_data, '*.npz')),
        out_data=out_data,
        max_samples=max_samples)

    # Main analysis: gamma-net versus seung models
    main_out_name = 'main_synth_ffn_experiment_data'
    main_model_batch = {
        'gammanet_t8_per_pixel': 'scarlet',
        'ffn_per_pixel': 'kelly green',
        'ffn_per_pixel_BSDS_init': 'darkgreen'}
    plot_wrapper(
        out_name=main_out_name,
        keep_models=main_model_batch,
        data_files=glob(os.path.join(jk_data, '*.npz')),
        out_data=out_data,
        max_samples=max_samples)

    # Main analysis: gamma-net versus seung models part 2
    main_out_name = 'extra_main_synth_experiment_data_2'
    main_model_batch = {
        'gammanet_t8_per_pixel': 'scarlet',
        'seung_unet_per_pixel': 'dusty blue',
        'seung_unet_per_pixel_BSDS_init': 'grey',
        # 'seung_unet_per_pixel_param_ctrl': 'aqua',
        'seung_unet_per_pixel_adabn': 'tiffany blue',
        'seung_unet_per_pixel_param_ctrl_rn_2': 'purple',
        'seung_unet_per_pixel_wd': 'lightish blue'}
    plot_wrapper(
        out_name=main_out_name,
        keep_models=main_model_batch,
        data_files=glob(os.path.join(jk_data, '*.npz')),
        out_data=out_data,
        max_samples=max_samples)

    # Compare gammanet to RNNs
    rnn_out_name = 'rnn_synth_experiment_data'
    rnn_model_batch = {
        'gammanet_t8_per_pixel': 'scarlet',
        # 'gru_v2_t8_per_pixel': 'light teal',
        # 'lstm_v2_t8_per_pixel': 'light sea green'}  # ,
        'hgru_bn_per_pixel': 'faded green',
        'gru_t8_per_pixel': 'dark teal',
        'lstm_t8_per_pixel': 'dark sea green'}
    plot_wrapper(
        out_name=rnn_out_name,
        keep_models=rnn_model_batch,
        data_files=glob(os.path.join(jk_data, '*.npz')),
        out_data=out_data,
        max_samples=max_samples)

    # Compare gammanet to lesioned gammanets
    lesion_gamma_out_name = 'lesion_gamma_synth_experiment_data'
    lesion_gamma_model_batch = {
        'gammanet_t8_per_pixel': 'scarlet',
        'gammanet_t4_per_pixel': 'red orange',
        'gammanet_t1_per_pixel': 'orange'}
    plot_wrapper(
        out_name=lesion_gamma_out_name,
        keep_models=lesion_gamma_model_batch,
        data_files=glob(os.path.join(jk_data, '*.npz')),
        out_data=out_data,
        max_samples=max_samples)

    # Extra lesions
    lesion_gamma_out_name = 'extra_lesion_gamma_synth_experiment_data'
    lesion_gamma_model_batch = {
        'gammanet_t8_per_pixel': 'scarlet',
        'gammanet_t8_per_pixel_disinhibition': 'dark pink',
        'hgru_bn_per_pixel': 'faded green',
        'gammanet_t8_per_pixel_skips': 'salmon',
        'gammanet_t8_per_pixel_lesion_mult': 'magenta',
        'gammanet_t8_per_pixel_lesion_add': 'purplish pink'}
    plot_wrapper(
        out_name=lesion_gamma_out_name,
        keep_models=lesion_gamma_model_batch,
        data_files=glob(os.path.join(jk_data, '*.npz')),
        out_data=out_data,
        max_samples=max_samples)

    # Plot everything
    all_out_name = 'all_models'
    all_model_batch = {
        'gammanet_t8_per_pixel': 'scarlet',
        'gammanet_t8_per_pixel_skips': 'salmon',
        'gammanet_t8_per_pixel_lesion_mult': 'magenta',
        'gammanet_t8_per_pixel_lesion_add': 'purplish pink',
        'gammanet_t4_per_pixel': 'red orange',
        'gammanet_t1_per_pixel': 'orange',
        'hgru_bn_per_pixel': 'faded green',
        'gru_t8_per_pixel': 'lightblue',
        'lstm_t8_per_pixel': 'french blue',
        'gru_v2_t8_per_pixel': 'light seafoam',
        'lstm_v2_t8_per_pixel': 'twilight', 
        'ffn_per_pixel': 'kelly green',
        'gammanet_t8_per_pixel_disinhibition': 'dark pink',
        'seung_unet_per_pixel_param_ctrl_rn_2': 'deep aqua',
        # 'seung_unet_per_pixel_param_ctrl': 'aqua',
        'seung_unet_per_pixel_wd': 'slate',
        'seung_unet_per_pixel_BSDS_init': 'deep sky blue'}
    plot_wrapper(
        out_name=all_out_name,
        keep_models=all_model_batch,
        data_files=glob(os.path.join(jk_data, '*.npz')),
        out_data=out_data,
        max_samples=max_samples)

if __name__ == '__main__':
    main()

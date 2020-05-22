import os
import numpy as np
import pandas as pd
from glob import glob
from matplotlib import pyplot as plt
from tqdm import tqdm


def sigmoid_fun(x):
    """Apply sigmoid to maps before mAP."""
    return 1 / (1 + np.exp(x))


jk_data = 'data_to_process_for_jk'
out_name = 'processed_images_for_jk'
grouped_file = pd.read_csv(os.path.join(jk_data, 'main_synth_experiment_data', 'main_synth_experiment_data_grouped.csv'))
keep_models = ['gammanet_t8_per_pixel', 'seung_unet_per_pixel']
apply_sigmoid, glob_files = True, False

# Get file paths
if glob_files:
    image_files = glob(os.path.join(jk_data, '*.npz'))
    image_files = [x for x in image_files if out_name not in x]
else:
    image_files = grouped_file[
        pd.Series.isin(grouped_file.model, keep_models)].file_name.as_matrix()
sel_image, sel_dim, data_block = 0, 0, 0

val_images, val_labels, val_logits = [], [], []
model_names = []
dm, train_datasets, val_datasets = [], [], []
for idx, f in tqdm(
        enumerate(image_files),
        total=len(image_files),
        desc='Gathering data for JK'):
    try:
        d = np.load(f)
        data_dict = d['val_dict'][data_block]
        config = d['config'].item()
        maps = d['maps']
        map_len = len(maps)
        dm += [np.vstack([
            [idx] * map_len,
            maps,
            [config.train_dataset] * map_len,
            [config.val_dataset] * map_len,
            [config.model] * map_len
        ])]
        val_images += [data_dict['images'][sel_image].squeeze()]
        val_labels += [data_dict['labels'][sel_image].squeeze()]
        model_names += [config.model]
        train_datasets += [config.train_dataset]
        val_datasets += [config.val_dataset]
        if apply_sigmoid:
            val_logits += [sigmoid_fun(
                data_dict['logits'][sel_image][..., -2].squeeze())]
        else:
            val_logits += [data_dict['logits'][sel_image].squeeze()]
    except Exception:
        print 'Failed on %s' % f
val_images = np.array(val_images)
val_labels = np.array(val_labels)
val_logits = np.array(val_logits)
out_path = os.path.join(jk_data, out_name)
df = pd.DataFrame(
    np.concatenate(dm, axis=1).transpose(),
    columns=[
        'model_idx',
        'average_precision',
        'train_dataset',
        'val_dataset',
        'model'])
np.savez(
    out_path,
    val_images=val_images,
    val_labels=val_labels,
    val_logits=val_logits,
    average_precision_data=df)
print('Saved JK data to: %s' % out_path)

# Make plots for all combinations of train/val
model_names = np.array(model_names)
train_datasets = np.array(train_datasets)
val_datasets = np.array(val_datasets)
idxs = np.arange(len(val_images) // 2).repeat(2)
for i in range(len(np.unique(idxs))):
    f = plt.figure(dpi=450)
    idx  = (i == idxs).ravel()
    ims = val_images[idx]
    labs = val_labels[idx]
    logs = val_logits[idx]
    mns = model_names[idx]
    tr = train_datasets[idx]
    va = val_datasets[idx]
    plt.subplot(3, 2, 1)
    plt.imshow(ims[0], cmap='Greys')
    plt.axis('off')
    plt.subplot(3, 2, 2)
    plt.imshow(labs[0], cmap='Greys_r')
    plt.axis('off')
    # plt.subplot(3, 2, 3)
    # plt.imshow(ims[1], cmap='Greys_r')
    # plt.axis('off')
    # plt.subplot(3, 2, 4)
    # plt.imshow(labs[1], cmap='Greys_r')
    plt.axis('off')
    plt.subplot(3, 2, 5)
    plt.imshow((logs[0] > 0.5).astype(np.float32), cmap='Greys_r', vmin=0, vmax=1)
    plt.axis('off')
    plt.subplot(3, 2, 6)
    plt.imshow((logs[1] > 0.5).astype(np.float32), cmap='Greys_r', vmin=0, vmax=1)
    plt.axis('off')
    plt.savefig(os.path.join(jk_data, '%s_%s.pdf' % (tr[0], va[0])))
    plt.close(f)

os._exit(1)

plt.subplot(2,4,1);plt.imshow(val_labels[0]);plt.subplot(2,4,2);plt.imshow(val_labels[1]);plt.subplot(2,4,3);plt.imshow(val_labels[6]);plt.subplot(2,4,4);plt.imshow(val_labels[7]);plt.subplot(2,4,5);plt.imshow(val_labels[12]);plt.subplot(2,4,6);plt.imshow(val_labels[13]);plt.subplot(2,4,7);plt.imshow(val_labels[18]);plt.subplot(2,4,8);plt.imshow(val_labels[19]);



# Plot stuff
gammanet_index = [
    idx
    for idx in range(len(model_names))
    if 'fgru' in model_names[idx]][0]
unet_index = [
    idx
    for idx in range(len(model_names))
    if 'seung_unet_per_pixel' == model_names[idx]][0]
unet_wd_index = [
    idx
    for idx in range(len(model_names))
    if 'seung_unet_per_pixel_wd' == model_names[idx]][0]
unet_small_index = [
    idx
    for idx in range(len(model_names))
    if 'seung_unet_per_pixel_param_ctrl' == model_names[idx]][0]\

f = plt.figure(dpi=300)
plt.subplot(242)
plt.imshow(val_images[0])
plt.axis('off')
plt.title('Validation image')
plt.subplot(243)
plt.imshow(val_labels[0], cmap='Greys_r')
plt.axis('off')
plt.title('Validation label')
plt.subplot(245)
plt.imshow(
    np.round(val_logits[unet_index]), cmap='Greys_r', vmin=0, vmax=1)
plt.axis('off')
plt.title('UNet')
plt.subplot(246)
plt.imshow(
    np.round(val_logits[unet_wd_index]), cmap='Greys_r', vmin=0, vmax=1)
plt.axis('off')
plt.title('UNet w/ weight decay')
plt.subplot(247)
plt.imshow(
    np.round(val_logits[unet_small_index]), cmap='Greys_r', vmin=0, vmax=1)
plt.axis('off')
plt.title('Small UNet parameter matched to $gamma$-net')
plt.subplot(248)
plt.imshow(
    np.round(val_logits[gammanet_index]), cmap='Greys_r', vmin=0, vmax=1)
plt.axis('off')
plt.title('$gamma$-net')
plt.savefig(os.path.join(jk_data, 'membrane_figs.pdf'))
plt.show()


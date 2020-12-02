import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import rotate
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_curve(activity, moments, means, stds, clf):
    # Normalize activities
    sel_units = (activity - means) / stds

    # Map responses
    inv_clf = np.linalg.inv(clf.T.dot(clf))
    inv_matmul = inv_clf.dot(clf.T)
    sel_units = inv_matmul.dot(sel_units.T)
    return sel_units


moments_file = "../undo_bias/neural_models/linear_moments/INSILICO_BSDS_vgg_gratings_simple_tb_feature_matrix.npz"
model_file = "../undo_bias/neural_models/linear_models/INSILICO_BSDS_vgg_gratings_simple_tb_model.joblib.npy"
moments = np.load(moments_file)
means = moments["means"]
stds = moments["stds"]
clf = np.load(model_file).astype(np.float32)
version = "exc"

all_args, all_diffs, all_peaks = [], [], []
thetas = [1, 31, 61, 91, 121, 151]
for theta in thetas:
    target = np.load("perturb_viz/gammanet_full_orientation_probe_outputs_data.npy")[:, theta]
    data = np.load("circuits_BSDS_{}_perturb/BSDS_{}_perturb_circuit_circuit_{}_{}_optim.npy".format(version, version, version, theta))[0][0]
    data = data.squeeze()
    mask = data.mean(-1) != 1
    he, wi = np.where(mask)
    args = np.zeros((mask.shape[0], mask.shape[1]))
    diffs = np.zeros_like(args)
    peaks = np.zeros_like(args)
    for h, w in zip(he, wi):
        activities = data[h:h + 4, w: w + 4]
        activities = activities.reshape(1, -1)
        tc = get_curve(activities, moments, means, stds, clf)
        args[h, w] = np.abs(np.argmax(tc) - np.argmax(target))
        diffs[h, w] = np.mean(np.abs(tc.squeeze() - target))
        # diffs[h, w] = np.mean((tc.squeeze() - target) ** 2)
        peaks[h, w] = np.abs(tc.squeeze()[np.round(theta / 30).astype(int)] - target[np.round(theta / 30).astype(int)])
    all_args.append(rotate(args, theta))
    all_diffs.append(rotate(diffs, theta))
    all_peaks.append(rotate(peaks, theta))

f, axs = plt.subplots(1, 3, figsize=(16, 4))
im0 = axs[0].imshow(np.stack(all_args).mean(0), cmap="RdBu")
axs[0].set_xticks([])
axs[0].set_yticks([])
divider = make_axes_locatable(axs[0])
cax = divider.append_axes('right', size='5%', pad=0.05)
f.colorbar(im0, cax=cax, orientation='vertical')

im1 = axs[1].imshow(np.stack(all_diffs).mean(0)[25:-25, 25:-25], cmap="RdBu")
axs[1].set_xticks([])
axs[1].set_yticks([])
divider = make_axes_locatable(axs[1])
cax = divider.append_axes('right', size='5%', pad=0.05)
f.colorbar(im1, cax=cax, orientation='vertical')

im2 = axs[2].imshow(np.stack(all_peaks).mean(0), cmap="RdBu")
axs[2].set_xticks([])
axs[2].set_yticks([])
divider = make_axes_locatable(axs[2])
cax = divider.append_axes('right', size='5%', pad=0.05)
f.colorbar(im2, cax=cax, orientation='vertical')

plt.show()


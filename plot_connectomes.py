import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pylab as pl
from glob import glob
import os
from argparse import ArgumentParser
from matplotlib.font_manager import FontProperties
from skimage.transform import rotate
from scipy.signal import convolve


def process_h(hiddens, model):
    """Process the hidden state using model weights to flip signs if needed."""
    # h2 bn via c1_bn_gamma_ and c1_bn_beta_
    nh = (hiddens - hiddens.mean(axis=(1, 2), keepdims=True)) / hiddens.std(axis=(1, 2), keepdims=True)
    nh = model["c1_bn_beta_0"] + nh * model["c1_bn_gamma_0"]

    # c1 = h2 conv with horizontal_kernels_inh
    c1 = convolve(nh, model["symm_horizontal_kernels_inh_0"], mode="same")

    # Sup-drive = (alpha * h2 + mu) * c1
    drive = (model["alpha_0"] * nh + model["recurrent_vgg16_mu"]) * c1
    return drive


def main(path, channel=0):
    """Plot connectomes."""
    connectome_files = glob(path)
    performance_files = glob(path.replace("optim.npy", "perf.npy"))
    tuning_exp_files = glob(path.replace("optim.npy", "curves.npy"))
    tuning_gt_files = glob(path.replace("optim.npy", "label.npy"))
    model_vars = np.load("gn_vars.npz")
    assert len(connectome_files), "Couldnt find anythin {}".format(path)
    connectome_files = sorted(connectome_files, key=os.path.getmtime)
    performance_files = sorted(performance_files, key=os.path.getmtime)
    tuning_exp_files = sorted(tuning_exp_files, key=os.path.getmtime)
    tuning_gt_files = sorted(tuning_gt_files, key=os.path.getmtime)
    results_dir = path.split(os.path.sep)[0]

    # How many distinct bins have we finished?
    bins = np.asarray([int(x.split("_")[-2].replace(".py", "")) for x in connectome_files])
    num_bins = len(bins)

    # Plot performance + tuning in subplots on the same figure
    f, axs = plt.subplots(3, num_bins, figsize=(18, 5))  # noqa perf/tuning, num bins  TODO: Add images
    plt.subplots_adjust(wspace=0.7, hspace=0.3)
    axs = axs.ravel()
    perf_tuning = zip(performance_files, tuning_exp_files, tuning_gt_files)
    fontP = FontProperties()
    fontP.set_size('xx-small')
    rcs = []
    for idx, pt in enumerate(perf_tuning):
        perf, tex, tgt = pt

        # Plot performance
        ax = axs[idx]
        perf = np.load(perf).squeeze()
        ax.plot(perf, "k", linewidth=1)
        ax.set_xlabel("Optim steps", fontsize=6)
        ax.set_title("${}^\circ$".format(bins[idx] - 91), fontsize=6)  # noqa
        ax.set_ylabel("$L^2$ loss", fontsize=6)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.tick_params(axis='both', which='minor', labelsize=6)

        # Plot tuning
        ax = axs[idx + num_bins]
        gt = np.load(tgt).squeeze()[-6:]
        ex = np.load(tex).squeeze()[:, :6]
        n = ex.shape[0]
        colors = pl.cm.RdBu_r(np.linspace(0, 1, n))
        for i in range(n):
            ax.plot(np.concatenate((ex[i], [ex[i][0]])), color=colors[i], label=i)
        ax.plot(np.concatenate((gt, [gt[0]])), "k--", label="GT")
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.tick_params(axis='both', which='minor', labelsize=6)
        ax.set_xlabel("Orientation", fontsize=6)
        ax.set_xticks(np.arange(7))
        ax.set_xticklabels([-90, -60, -30, 0, 30, 60, 90], fontsize=6)
        ax.set_ylabel("Population response", fontsize=6)
        # ax.legend(
        #     loc="upper left",
        #     prop=fontP,
        #     fancybox=True,
        #     shadow=False)

        # Plot connectome
        ax = axs[idx + num_bins * 2]
        con = np.load(connectome_files[idx])[0][channel]  # [0] - np.load(connectome)[channel][1]  # additive/mulitiplicative
        con = con.squeeze().mean(-1)
        minmax = max(np.abs(con[40:45, 40:45].min()), con[40:45, 40:45].max())
        minmax = minmax + minmax * 2
        ax.imshow(con, cmap="RdBu_r", vmin=minmax * np.sign(con.min()), vmax=minmax)
        ax.set_xticks([])
        ax.set_yticks([])
        theta = bins[idx]
        ax.set_xlabel(r"$\theta = {}$".format(theta))
        rcs.append(rotate(con, bins[idx], preserve_range=True, order=0))
    plt.savefig(os.path.join(results_dir, "performance.pdf"))
    plt.show()
    plt.close(f)

    """
    # Plot mean connectome for every theta
    f, axs = plt.subplots(1, len(connectome_files), figsize=(15, 5))
    # if not isinstance(axs, list) or not isinstance(axs, np.ndarray):
    #     axs = [axs]
    rcs = []
    for idx, (ax, connectome, theta) in enumerate(zip(axs, connectome_files, bins)):
        con = np.load(connectome)[0][channel]  # [0] - np.load(connectome)[channel][1]  # additive/mulitiplicative
        # con = process_h(hiddens=con, model=model_vars)
        con = con.squeeze().mean(-1)
        # minmax = max(-np.percentile(con, 5), np.percentile(con, 95))
        # minmax = max(np.abs(con[30:50, 30:50].min()), con[30:50, 30:50].max())
        minmax = max(np.abs(con[40:45, 40:45].min()), con[40:45, 40:45].max())
        minmax = minmax + minmax * 2
        # minmax = max(np.abs(con[40:45, 40:45].min()), con[40:45, 40:45].max())
        # minmax = np.percentile(con, 90)
        # ax.axis("off")
        ax.imshow(con, cmap="RdBu", vmin=minmax * np.sign(con.min()), vmax=minmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(r"$\theta = {}$".format(theta))
        # rcs.append(con)
        # rc += rotate(con, bins[idx], preserve_range=True, order=0)
        rcs.append(rotate(con, bins[idx], preserve_range=True, order=0))
    plt.savefig(os.path.join(results_dir, "connectomes.pdf"))
    plt.show()
    plt.close(f)
    """

    f = plt.figure(dpi=200)
    # rc = rc / (idx + 1)
    rcs = np.asarray(rcs)
    # mx = rcs.max()
    # mi = rcs.min()
    # rc = (rcs - mi) / mx
    rc = rcs.mean(0)
    minmax = max(np.abs(rc[40:45, 40:45].min()), rc[40:45, 40:45].max())
    if "circuit_exc" in path:
        minmax = minmax + minmax * 1
    elif "plaid_exc" in path:
        minmax = minmax + minmax
    else:
        minmax = minmax + minmax * 1
    # minmax = max(np.abs(rc.min()), rc.max())
    plt.imshow(rc, cmap="RdBu_r", vmin=minmax * np.sign(con.min()), vmax=minmax)
    plt.colorbar()
    plt.axis("off")
    plt.savefig(os.path.join(results_dir, "mean_connectome.pdf"))
    plt.show()
    plt.close(f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        dest="path",
        default="circuits_BSDS_exc_perturb/BSDS_exc_perturb_circuit_circuit_exc_*_optim.npy",
        # default="circuits_BSDS_inh_perturb/BSDS_inh_perturb_circuit_circuit_inh_*_optim.npy",

        # default="circuits_BSDS_exc_perturb/exc_1.5/BSDS_exc_perturb_circuit_circuit_exc_*_optim.npy",
        # default="circuits_BSDS_exc_perturb/exc_2.0/BSDS_exc_perturb_circuit_circuit_exc_*_optim.npy",
        # default="circuits_BSDS_inh_perturb/inh_0.001/BSDS_inh_perturb_circuit_circuit_inh_*_optim.npy",
        # default="circuits_BSDS_inh_perturb/inh_0.01/BSDS_inh_perturb_circuit_circuit_inh_*_optim.npy",
        # default="circuits_BSDS_inh_perturb/inh_0.1/BSDS_inh_perturb_circuit_circuit_inh_*_optim.npy",
        # default="circuits_BSDS_exc_perturb/BSDS_exc_perturb_circuit_circuit_exc_full_field_*_optim.npy",
        # default="circuits_BSDS_inh_perturb/BSDS_inh_perturb_circuit_circuit_inh_full_field_*_optim.npy",
        # default="circuits_BSDS_exc_phase_perturb/BSDS_exc_phase_perturb_circuit_circuit_exc_*_optim.npy",
        # default="circuits_BSDS_exc_phase_perturb_phase_180/BSDS_exc_phase_perturb_circuit_circuit_exc_*_optim.npy",
        # default="circuits_BSDS_inh_phase_perturb/BSDS_inh_phase_perturb_circuit_circuit_inh_*_optim.npy",
        # default="circuits_BSDS_inh_perturb/BSDS_inh_perturb_circuit_circuit_inh_*_optim.npy",
        # default="circuits_BSDS_exc_perturb/BSDS_exc_perturb_circuit_circuit_plaid_exc_*_optim.npy",
        # default="circuits_BSDS_inh_perturb/BSDS_inh_perturb_circuit_circuit_plaid_inh_*_optim.npy",
        help="Name of experiment with model responses.")
    main(**vars(parser.parse_args()))


import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pylab as pl


def tuning_weighted_circuit(x):
    w = (x ** 2).sum(0).sum(0)[None, None]                                                                                                                                
    w = w / w.sum()                                                                                                                                                       
    return (x * w).sum(-1)


i = np.load("exc_perturbs/optim.npy")  # .mean(2)
e = np.load("inh_perturbs/optim.npy")  # .mean(2)
# diff = (e - i).squeeze().mean(-1)
# diff = (((e-e.mean(2, keepdims=True)) / e.std(2, keepdims=True)) - ((i-i.mean(2, keepdims=True)) / i.std(2, keepdims=True))).mean(2).squeeze()
# i = i.squeeze().mean(-1)
# e = e.squeeze().mean(-1)
i = tuning_weighted_circuit(i.squeeze())
e = tuning_weighted_circuit(e.squeeze())
diff = i - e
diff_max = max(diff.max(), -diff.min())
i_max = max(i.max(), -i.min())
e_max = max(e.max(), -e.min())

overall_max = max(i_max, e_max)

f = plt.figure()
plt.subplot(131);plt.title("Inhibition");plt.imshow(-i.squeeze(), cmap="RdBu", vmax=overall_max, vmin=-overall_max)
plt.subplot(132);plt.title("Excitation");plt.imshow(e.squeeze(), cmap="RdBu_r", vmax=overall_max, vmin=-overall_max)
plt.subplot(133);plt.title("E - I");plt.imshow(diff, vmin=-diff_max, vmax=diff_max, cmap="RdYlGn_r");plt.colorbar();plt.show()
plt.close(f)

f, ax = plt.subplots(1, 1)
i_curves = np.load("inh_perturbs/curves.npy").squeeze()
e_curves = np.load("exc_perturbs/curves.npy").squeeze()
inh_labels = np.load("inh_perturbs/label.npy").squeeze()
exc_labels = np.load("exc_perturbs/label.npy").squeeze()
n = i_curves.shape[0]
colors = pl.cm.Blues(np.linspace(0, 1, n))
i_curves = np.concatenate((i_curves, i_curves[:, 0][:, None]), 1)
[plt.plot(i_curves[i], label="{}".format(i), color=colors[i]) for i in range(i_curves.shape[0])]
plt.title("Inh perturb recovery over epochs of optim")
inh_labels = np.concatenate((inh_labels, [inh_labels[-1]]))
plt.plot(inh_labels, "k--", label="Target")
plt.legend()
plt.ylabel("Neural response")
plt.xlabel("Orientation")
ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
ax.set_xticklabels([-90, -60, -30, 0, 30, 60, 90])
plt.show()
plt.close(f)
f, ax = plt.subplots(1, 1)
n = e_curves.shape[0]
e_curves = np.concatenate((e_curves, e_curves[:, 0][:, None]), 1)
colors = pl.cm.Reds(np.linspace(0, 1, n))
[plt.plot(e_curves[i], label="{}".format(i), color=colors[i]) for i in range(e_curves.shape[0])]
plt.title("Exc perturb recovery over epochs of optim")
exc_labels = np.concatenate((exc_labels, [exc_labels[-1]]))
plt.plot(exc_labels, "k--", label="Target")
plt.legend()
plt.ylabel("Neural response")
plt.xlabel("Orientation")
ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
ax.set_xticklabels([-90, -60, -30, 0, 30, 60, 90])
plt.show()
plt.close(f)

f = plt.figure()
plt.title("Functional connectivity of BSDS-trained $\gamma$-net (fGRU-0)")
plt.imshow(diff, vmin=-diff_max, vmax=diff_max, cmap="RdBu_r")
plt.axis("off")
co = plt.colorbar()
co.set_label("(-) Inhibition                  Excitation (+)")
plt.show()
plt.close(f)



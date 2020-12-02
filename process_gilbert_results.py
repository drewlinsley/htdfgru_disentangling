import os
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from glob import glob
from sklearn.metrics import average_precision_score, log_loss
import pandas as pd


sigmoid = lambda x: 1 / (1 + np.exp(-x))
dirs = glob("gilbert_length*_shear*")
all_logits, all_labels = [], []
lengths, shears = [], []
experiment = []
means = []
exp = []
for idx, d in tqdm(enumerate(dirs), total=len(dirs)):
    length = int(d.split("length")[-1].split("_")[0])
    shear = d.split("_")[-1].replace("shearp", "").replace("shear", "")
    if "-" in shear:
        shear = -float(shear.strip("-").split("_")[-1])
    else:
        shear = float(shear.split("_")[-1])
    print("Length: {} | Shear: {}".format(length, shear))
    lengths.append(length)
    shears.append(shear)
    files = glob(os.path.join(d, "*.npz"))
    data = np.load(files[0])["test_dict"]
    logits, labels = [], []
    for da in data:
        logits.append(da["logits"])
        labels.append(da["labels"])
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    accuracy = np.mean((logits > 0).astype(np.float32) == labels)
    ap = average_precision_score(y_true=labels.reshape(-1, 1), y_score=logits.reshape(-1, 1))
    loss = log_loss(y_true=labels.reshape(-1, 1), y_pred=sigmoid(logits).reshape(-1, 1))
    all_logits.append(logits)
    all_labels.append(labels)
    # means.append(loss)  # accuracy
    means.append(accuracy)
    exp.append(idx)
    experiment.append(np.zeros_like(logits) + idx)
means = np.asarray(means).reshape(-1, 1)
lengths = np.asarray(lengths).reshape(-1, 1)
shears = np.asarray(shears).reshape(-1, 1)
df = pd.DataFrame(np.concatenate((1 - means, lengths, shears), -1), columns=["scores", "lengths", "relative spacing"])

f, axs = plt.subplots(1, 1, figsize=(6, 4))
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.lineplot(data=df, x="lengths", y="scores", hue="relative spacing", palette=sns.color_palette("husl", 5), alpha=0.75, linewidth=3, solid_joinstyle="miter", marker="o")
plt.ylabel("1 - loss (higher=better)")
plt.xlabel("Contour length")
plt.show()
plt.close(f)


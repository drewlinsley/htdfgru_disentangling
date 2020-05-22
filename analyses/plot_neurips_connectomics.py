"""Plot connectomics for neurips sub."""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.style as style
import matplotlib


f = 'maps/neurips_connectomics.csv'
df = pd.read_csv(f)
# style.use('ggplot')
# matplotlib.rcParams['font.family'] = "serif"
sns.set_context('talk')


# Recode augmented as 120
df['condition'][df['condition'] == 'augmented'] = 120
df['condition'] = pd.to_numeric(df['condition'])
df['map'] = pd.to_numeric(df['map'])
pallette = {
    'gammanet': 'scarlet',
    'UNet-BN': 'dusty blue',
    'UNet-IN': 'grey'}

# Plot two ways. First Seung Unet(BN) vs. Gammanet
compare_df = df[df['model'] != 'UNet-IN']
f = plt.figure()
g = sns.factorplot(
    data=compare_df,
    x='condition',
    y='map',
    hue='model',
    col='dataset',
    sharey=True,
    sharex=True,
    margin_titles=True,
    pallette=pallette)
plt.ylim([0., 1])
plt.savefig('seung_vs_gn.pdf', dpi=150)
# plt.show()
plt.close(f)

# Second, Seung Unet(BN) vs. Seung Unet(IN) vs. Gammanet
f = plt.figure()
g = sns.factorplot(
    data=df,
    x='condition',
    y='map',
    hue='model',
    col='dataset',
    sharey=True,
    sharex=True,
    margin_titles=True,
    pallette=pallette)
plt.ylim([0., 1])
plt.savefig('seung_vs_in_vs_gn.pdf', dpi=150)
# plt.show()
plt.close(f)

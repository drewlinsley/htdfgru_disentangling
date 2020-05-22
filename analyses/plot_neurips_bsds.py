"""Plot connectomics for neurips sub."""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.style as style
import matplotlib


f = 'neurips_data/maps/neurips_bsds.csv'
df = pd.read_csv(f)
# style.use('ggplot')
# matplotlib.rcParams['font.family'] = "serif"
sns.set_context('talk')


# Recode augmented as 120
df['condition'] = pd.to_numeric(df['condition'])
df['model'][df['model'] == 'gammanet'] = 0
df['model'][df['model'] == 'BDCN'] = 1
df['map'] = pd.to_numeric(df['map'])
df['model'] = pd.to_numeric(df['model'])
pallette = {
    1: 'scarlet',
    0: 'dusty blue'}

# Plot two ways. First Seung Unet(BN) vs. Gammanet
f = plt.figure()
g = sns.factorplot(
    data=df,
    x='condition',
    y='map',
    hue='model',
    color='firebrick',
    col='dataset',
    sharey=True,
    sharex=True,
    margin_titles=True,
    pallette=pallette)

plt.ylim([0.5, 1])
plt.savefig('bsds_perf_fake.pdf', dpi=150)
plt.show()
plt.close(f)


# Plot two ways. First Seung Unet(BN) vs. Gammanet
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
plt.ylim([0.5, 1])
plt.savefig('bsds_perf_real.pdf', dpi=150)
plt.show()
plt.close(f)

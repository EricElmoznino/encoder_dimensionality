import seaborn as sns
from matplotlib import pyplot as plt


def plot_eigenspectrum(data, x, y, hue=None, ax=None, log_scale=True, **kwargs):
    ax = sns.lineplot(data=data, x=x, y=y, hue=hue, ax=ax, **kwargs)
    ax.set_xlim(data[x].min(), data[x].max())
    ax.set_ylim(data[y].min(), data[y].max())
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    return ax

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


def plot_correlations(data, x, y, hue, grid=None, plt_kwargs=None, grid_kwargs=None):
    if plt_kwargs is None:
        plt_kwargs = {}
    if grid_kwargs is None:
        grid_kwargs = {}

    if grid is None:
        ax = sns.scatterplot(data=data, x=x, y=y, hue=hue, **plt_kwargs)
        ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        return ax
    else:
        g = sns.FacetGrid(data=data, hue=hue, col=grid, **grid_kwargs)
        g.map(sns.scatterplot, x, y, **plt_kwargs)
        return g

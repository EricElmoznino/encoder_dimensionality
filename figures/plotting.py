import math
from scipy import stats
import scipy.cluster.hierarchy as spc
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch


def annotate_corr_grid(g, x, y, method='pearson'):
    g.map_dataframe(annotate_corr, x=x, y=y, method=method)


def annotate_corr(data, x, y, ax=None, method='spearman', **kwargs):
    if method == 'pearson':
        r, p = stats.pearsonr(data[x], data[y])
    elif method == 'spearman':
        r, p = stats.spearmanr(data[x], data[y])
    else:
        raise NotImplementedError()
    if ax is None:
        ax = plt.gca()
    ax.text(.05, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
            transform=ax.transAxes)


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
        annotate_corr(data, x=x, y=y, ax=ax)
        return ax
    else:
        g = sns.FacetGrid(data=data, hue=hue, col=grid, **grid_kwargs)
        g.map(sns.scatterplot, x, y, **plt_kwargs)
        annotate_corr_grid(g, x=x, y=y)
        return g


def plot_metrics_categorical(kind, data, x, ys, labels=None, col_wrap=5,
                             fig_kwargs=None, plt_kwargs=None, grid_kwargs=None):
    if fig_kwargs is None:
        fig_kwargs = {}
    if plt_kwargs is None:
        plt_kwargs = {}
    if grid_kwargs is None:
        grid_kwargs = {}

    x_order = data[x].unique() if x is not None else None
    ncols = min(len(ys), col_wrap)
    nrows = math.ceil(len(ys) / col_wrap)
    fig = plt.figure(**fig_kwargs)
    gs = GridSpec(nrows=nrows, ncols=ncols, **grid_kwargs)

    axs = []
    for i, y in enumerate(ys):
        row, col = i // ncols, i % ncols
        ax = fig.add_subplot(gs[row, col])
        axs.append(ax)

        if kind == 'bar':
            sns.barplot(data=data, x=x, y=y, ax=ax, order=x_order, **plt_kwargs)
        elif kind == 'violin':
            sns.violinplot(data=data, x=x, y=y, ax=ax, order=x_order, **plt_kwargs)
        elif kind == 'box':
            sns.boxplot(data=data, x=x, y=y, ax=ax, order=x_order, **plt_kwargs)
        elif kind == 'strip':
            sns.stripplot(data=data, x=x, y=y, ax=ax, order=x_order, **plt_kwargs)

        ax.set(xticklabels=[], xlabel=None, ylabel=None)
        if labels is None:
            ax.set_title(y)
        else:
            ax.set_title(labels[i])

    if x is not None:
        patches = [Patch(color=sns.color_palette()[i], label=t) for i, t in enumerate(x_order)]
        fig.legend(handles=patches, loc='upper left', bbox_to_anchor=(1.05, 0.65))

    fig.tight_layout()

    return fig, axs


def plot_metrics_trend(data, x, ys, hue, labels=None, col_wrap=5,
                       fig_kwargs=None, plt_kwargs=None, grid_kwargs=None):
    if fig_kwargs is None:
        fig_kwargs = {}
    if plt_kwargs is None:
        plt_kwargs = {}
    if grid_kwargs is None:
        grid_kwargs = {}

    ncols = min(len(ys), col_wrap)
    nrows = math.ceil(len(ys) / col_wrap)
    fig = plt.figure(**fig_kwargs)
    gs = GridSpec(nrows=nrows, ncols=ncols, **grid_kwargs)

    axs = []
    for i, y in enumerate(ys):
        row, col = i // ncols, i % ncols
        ax = fig.add_subplot(gs[row, col])
        axs.append(ax)

        sns.lineplot(data=data, x=x, y=y, hue=hue, ax=ax, **plt_kwargs)

        ax.set(xticklabels=[], xlabel=None, ylabel=None)
        if labels is None:
            ax.set_title(y)
        else:
            ax.set_title(labels[i])
        ax.legend().remove()

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(1.05, 0.65))

    fig.tight_layout()

    return fig, axs


def plot_metrics_vs_other(data, xs, y, hue, labels=None, col_wrap=5,
                          fig_kwargs=None, plt_kwargs=None, grid_kwargs=None):
    if fig_kwargs is None:
        fig_kwargs = {}
    if plt_kwargs is None:
        plt_kwargs = {}
    if grid_kwargs is None:
        grid_kwargs = {}

    ncols = min(len(xs), col_wrap)
    nrows = math.ceil(len(xs) / col_wrap)
    fig = plt.figure(**fig_kwargs)
    gs = GridSpec(nrows=nrows, ncols=ncols, **grid_kwargs)

    axs = []
    for i, x in enumerate(xs):
        row, col = i // ncols, i % ncols
        ax = fig.add_subplot(gs[row, col])
        axs.append(ax)

        sns.scatterplot(data=data, x=x, y=y, hue=hue, ax=ax, **plt_kwargs)

        ax.set(xlabel=None, ylabel=None)
        if labels is None:
            ax.set_title(x)
        else:
            ax.set_title(labels[i])
        ax.legend().remove()

        annotate_corr(data, x=x, y=y, ax=ax)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(1.05, 0.65))

    fig.tight_layout()

    return fig, axs


def plot_corr_heatmap(data, ax, method='pearson', reorder=True):
    c = data.corr(method=method)
    if reorder:
        linkage = spc.linkage(c, method='centroid')
        z = spc.dendrogram(linkage, no_plot=True)
        idx = z['leaves']
        c = c.iloc[idx, :].iloc[:, idx]
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    ax = sns.heatmap(c, vmin=-1, vmax=1, cmap=cmap, square=True,
                     xticklabels=True, yticklabels=True, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    return ax

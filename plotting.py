import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def connected_stripplot(data, x, y, group_id, ax=None, distribution='violin', jitter=0.05, xorder=None,
                        point_kwargs=None, distribution_kwargs=None, scatter_kwargs=None, line_kwargs=None):
    point_kwargs_defaults = {'color': sns.color_palette('Blues')[-1], 'errwidth': 1.5, 'capsize': 0.03}
    distribution_kwargs_defaults = {'color': sns.color_palette('Blues')[0]}
    scatter_kwargs_defaults = {'color': 'darkgrey', 'markersize': 5}
    line_kwargs_defaults = {'color': 'darkgrey', 'linewidth': 0.5, 'linestyle': '-'}
    point_kwargs = merge_dict(point_kwargs_defaults, point_kwargs)
    distribution_kwargs = merge_dict(distribution_kwargs_defaults, distribution_kwargs)
    scatter_kwargs = merge_dict(scatter_kwargs_defaults, scatter_kwargs)
    line_kwargs = merge_dict(line_kwargs_defaults, line_kwargs)

    data = data[[x, y, group_id]]

    # Point plot
    ax = sns.pointplot(data=data, x=x, y=y, ax=ax, order=xorder,
                       **point_kwargs)
    plt.setp(ax.lines, zorder=102)
    plt.setp(ax.collections, zorder=102, label="")

    # Distribution plot plot
    if distribution == 'violin':
        ax = sns.violinplot(data=data, x=x, y=y, ax=ax, order=xorder,
                            cut=0, inner=None, **distribution_kwargs)
    elif distribution == 'box':
        ax = sns.boxplot(data=data, x=x, y=y, ax=ax, order=xorder,
                         fliersize=0, **distribution_kwargs)

    # Format data for strip and line plots
    datapivot = data.pivot(index=group_id, columns=x, values=y)
    if xorder is not None:
        datapivot = datapivot[xorder]
    data_xjitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=datapivot.values.shape),
                                columns=datapivot.columns)
    data_xjitter += np.arange(len(datapivot.columns))

    # Strip plot
    for col in datapivot.columns:
        ax.plot(data_xjitter[col], datapivot[col], 'o',
                zorder=101, **scatter_kwargs)
    ax.set_xticks(range(len(datapivot.columns)))
    ax.set_xticklabels(datapivot.columns)
    ax.set_xlim(-0.5, len(datapivot.columns) - 0.5)

    # Line plot
    for idx in range(len(datapivot)):
        ax.plot(data_xjitter.iloc[idx].values, datapivot.iloc[idx].values,
                zorder=100, **line_kwargs)

    return ax


def test_connected_stripplot():
    df = pd.DataFrame({'model': list(range(100)) * 2,
                       'transform': ['z-scoring'] * 100 + ['none'] * 100,
                       'performance': np.random.normal(loc=3, size=(100,)).tolist() +
                                      np.random.normal(loc=5, size=(100,)).tolist()})
    connected_stripplot(data=df, x='transform', y='performance', group_id='model',
                        xorder=['none', 'z-scoring'])
    plt.show()


def merge_dict(first, second):
    merged = {k: v for k, v in first.items()}
    if second is None:
        return merged
    for k, v in second.items():
        merged[k] = v
    return merged

if __name__ == '__main__':
    test_connected_stripplot()

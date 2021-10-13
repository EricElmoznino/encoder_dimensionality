import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from activation_models.generators import get_activation_models
from custom_model_tools.manifold import ManifoldStatisticsImageNet21k, ManifoldStatisticsImageNet, \
    ManifoldStatisticsObject2Vec, ManifoldStatisticsMajajHong2015

from utils import timed


@timed
def main(dataset, data_dir, pooling, debug=False):
    manifold_statistics_df = pd.DataFrame()
    for model, layers in get_activation_models():
        manifold_statistics = get_manifold_statistics(dataset, data_dir, model, pooling)
        manifold_statistics.fit(layers)
        manifold_statistics_df = manifold_statistics_df.append(manifold_statistics.as_df())
        if debug:
            break
    if not debug:
        manifold_statistics_df.to_csv(f'results/manifolds|dataset:{dataset}|pooling:{pooling}.csv', index=False)


def get_manifold_statistics(dataset, data_dir, activations_extractor, pooling):
    if dataset == 'imagenet':
        return ManifoldStatisticsImageNet(activations_extractor=activations_extractor,
                                          pooling=pooling)
    elif dataset == 'imagenet21k':
        return ManifoldStatisticsImageNet21k(data_dir=data_dir,
                                             activations_extractor=activations_extractor,
                                             pooling=pooling)
    elif dataset == 'object2vec':
        return ManifoldStatisticsObject2Vec(data_dir=data_dir,
                                            activations_extractor=activations_extractor,
                                            pooling=pooling)
    elif dataset == 'majajhong2015':
        return ManifoldStatisticsMajajHong2015(activations_extractor=activations_extractor,
                                               pooling=pooling)
    else:
        raise ValueError(f'Unknown manifold dataset: {dataset}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute and store manifold statistics of models')
    parser.add_argument('--dataset', type=str,
                        choices=['imagenet', 'imagenet21k', 'object2vec', 'majajhong2015'],
                        help='Dataset of concepts for which to compute manifold statistics')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory containing stimuli')
    parser.add_argument('--no_pooling', dest='pooling', action='store_false',
                        help='Do not perform global max-pooling prior to computing the manifold statistics')
    parser.add_argument('--debug', action='store_true',
                        help='Just run a single model to make sure there are no errors')
    args = parser.parse_args()

    main(dataset=args.dataset, data_dir=args.data_dir, pooling=args.pooling, debug=args.debug)

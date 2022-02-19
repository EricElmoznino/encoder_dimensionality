import argparse
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import xarray as xr
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from activation_models.generators import get_activation_models
from custom_model_tools.projection_distances import ProjectionDistancesImageFolder, \
    ProjectionDistancesObject2Vec, ProjectionDistancesImageNet
from utils import timed


@timed
def main(dataset, data_dir, pooling, debug=False):
    proj_da = []
    for model, layers in get_activation_models():
        proj = get_projection(dataset, data_dir, model, pooling)
        proj.fit(layers)
        proj_da.append(proj.as_dataarray())
        if debug:
            break
    proj_da = xr.concat(proj_da, dim='identifier')
    proj_da = proj_da.reset_index('identifier')
    if not debug:
        proj_da.to_netcdf(f'results/projections|dataset:{dataset}|pooling:{pooling}.csv')


def get_projection(dataset, data_dir, activations_extractor, pooling):
    if dataset == 'imagenet':
        return ProjectionDistancesImageNet(activations_extractor=activations_extractor,
                                           pooling=pooling)
    elif dataset == 'imagenet21k':
        return ProjectionDistancesImageFolder(data_dir=data_dir,
                                              activations_extractor=activations_extractor,
                                              pooling=pooling,
                                              stimuli_identifier='imagenet21k')
    elif dataset == 'object2vec':
        return ProjectionDistancesObject2Vec(data_dir=data_dir,
                                             activations_extractor=activations_extractor,
                                             pooling=pooling)
    else:
        raise ValueError(f'Unknown pairwise projections dataset: {dataset}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute and pairwise class projections')
    parser.add_argument('--dataset', type=str,
                        choices=['imagenet', 'imagenet21k', 'object2vec'],
                        help='Dataset for which to compute pairwise class projections')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory containing stimuli')
    parser.add_argument('--no_pooling', dest='pooling', action='store_false',
                        help='Do not perform global max-pooling prior to computing the projections')
    parser.add_argument('--debug', action='store_true',
                        help='Just run a single model to make sure there are no errors')
    args = parser.parse_args()

    main(dataset=args.dataset, data_dir=args.data_dir,
         pooling=args.pooling, debug=args.debug)

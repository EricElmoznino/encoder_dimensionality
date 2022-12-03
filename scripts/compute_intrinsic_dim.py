import os
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from activation_models.generators import get_activation_models
from custom_model_tools.intrinsic_dim import IntrinsicDimImageNet21k, IntrinsicDimImageNet, \
    IntrinsicDimObject2Vec, IntrinsicDimMajajHong2015
from utils import timed


@timed
def main(dataset, data_dir, pooling, debug=False):
    save_path = f'results/intrinsic-dim|dataset:{dataset}|pooling:{pooling}.csv'
    if os.path.exists(save_path):
        print(f'Results already exists: {save_path}')
        return
    
    intrinsic_dim_df = pd.DataFrame()
    for model, layers in get_activation_models():
        intrinsic_dim = get_intrinsic_dim(dataset, data_dir, model, pooling)
        intrinsic_dim.fit(layers)
        intrinsic_dim_df = intrinsic_dim_df.append(intrinsic_dim.as_df())
        if debug:
            break
    if not debug:
        intrinsic_dim_df.to_csv(save_path, index=False)


def get_intrinsic_dim(dataset, data_dir, activations_extractor, pooling):
    if dataset == 'imagenet':
        return IntrinsicDimImageNet(activations_extractor=activations_extractor,
                                    pooling=pooling)
    elif dataset == 'imagenet21k':
        return IntrinsicDimImageNet21k(data_dir=data_dir,
                                       activations_extractor=activations_extractor,
                                       pooling=pooling)
    elif dataset == 'object2vec':
        return IntrinsicDimObject2Vec(data_dir=data_dir,
                                      activations_extractor=activations_extractor,
                                      pooling=pooling)
    elif dataset == 'majajhong2015':
        return IntrinsicDimMajajHong2015(activations_extractor=activations_extractor,
                                         pooling=pooling)
    else:
        raise ValueError(f'Unknown intrinsic dimensionality dataset: {dataset}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute and store intrinsic dimensionalities of models')
    parser.add_argument('--dataset', type=str,
                        choices=['imagenet', 'imagenet21k', 'object2vec', 'majajhong2015'],
                        help='Dataset of image categories for which to compute intrinsic dimensionalities')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory containing stimuli')
    parser.add_argument('--no_pooling', dest='pooling', action='store_false',
                        help='Do not perform global max-pooling prior to computing the intrinsic dimensionalities')
    parser.add_argument('--debug', action='store_true',
                        help='Just run a single model to make sure there are no errors')
    args = parser.parse_args()

    main(dataset=args.dataset, data_dir=args.data_dir, pooling=args.pooling, debug=args.debug)

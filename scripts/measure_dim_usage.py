import os
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from activation_models.generators import get_activation_models
from custom_model_tools.dim_usage import DimUsageMajajHongIT
from utils import timed


@timed
def main(bench, regression, pooling, debug=False):
    save_path = f'results/dimusage|benchmark:{bench}|reg:{regression}|pooling:{pooling}.csv'
    if os.path.exists(save_path):
        print(f'Results already exists: {save_path}')
        return

    dim_usage_df = pd.DataFrame()
    for model, layers in get_activation_models():
        dim_usage = get_dim_usage_class(bench, model, regression, pooling)
        dim_usage.fit(layers)
        dim_usage_df = dim_usage_df.append(dim_usage.as_df())
        if debug:
            break

    if not debug:
        dim_usage_df.to_csv(save_path, index=False)


def get_dim_usage_class(bench, activations_extractor, regression, pooling):
    if bench == 'majajhong2015':
        return DimUsageMajajHongIT(activations_extractor=activations_extractor,
                                   pooling=pooling, regression=regression)
    else:
        raise ValueError(f'Unknown dimensionality usage benchmark: {bench}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Measure the degree of PC usage in encoding models')
    parser.add_argument('--bench', type=str, default='majajhong2015',
                        choices=['majajhong2015'],
                        help='Neural benchmark dataset to fit')
    parser.add_argument('--regression', type=str, default='pls',
                        choices=['pls', 'lin', 'l2'],
                        help='Partial-least-squares or ordinary-least-squares for fitting')
    parser.add_argument('--no_pooling', dest='pooling', action='store_false',
                        help='Do not perform global max-pooling')
    parser.add_argument('--debug', action='store_true',
                        help='Just run a single model to make sure there are no errors')
    args = parser.parse_args()

    main(bench=args.bench, regression=args.regression, pooling=args.pooling, debug=args.debug)

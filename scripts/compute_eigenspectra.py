import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from activation_models.generators import get_activation_models
from custom_model_tools.eigenspectrum import ImageNetLayerEigenspectrum
from utils import timed


@timed
def main(pooling, debug=False):
    eigspec_df = pd.DataFrame()
    eig_metrics_df = pd.DataFrame()
    for model, layers in get_activation_models():
        eigspec = ImageNetLayerEigenspectrum(model, pooling=pooling)
        eigspec.fit(layers)

        eigspec_df = eigspec_df.append(eigspec.as_df())
        eig_metrics_df = eig_metrics_df.append(eigspec.metrics_as_df())

        if debug:
            break

    if not debug:
        eigspec_df.to_csv(f'results/eigspectra|pooling:{pooling}.csv', index=False)
        eig_metrics_df.to_csv(f'results/eigmetrics|pooling:{pooling}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute and store eigenspectra of models')
    parser.add_argument('--pooling', dest='pooling', action='store_true',
                        help='Perform global max-pooling prior to computing the eigenspectrum')
    parser.add_argument('--no_pooling', dest='pooling', action='store_false',
                        help='Do not perform global max-pooling prior to computing the eigenspectrum')
    parser.set_defaults(pooling=True)
    parser.add_argument('--debug', action='store_true',
                        help='Just run a single model to make sure there are no errors')
    args = parser.parse_args()

    main(pooling=args.pooling, debug=args.debug)

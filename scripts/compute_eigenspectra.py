import argparse
from time import time
import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from activation_models.generators import get_activation_models
from custom_model_tools.eigenspectrum import ImageNetLayerEigenspectrum


def main(pooling, debug):
    start_time = time()

    for model, layers in get_activation_models():
        eigspec = ImageNetLayerEigenspectrum(model, pooling=pooling)
        eigspec.fit(layers)

        if debug:
            break

    end_time = time()
    elapsed_time = str(datetime.timedelta(seconds=(end_time-start_time)))
    print(f'Total runtime: {elapsed_time}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute and store eigenspectra of models')
    parser.add_argument('--pooling', type=bool, default=True,
                        help='Whether or not to perform global max-pooling prior to computing the eigenspectrum')
    parser.add_argument('--debug', action='store_true',
                        help='Just run a single model to make sure there are no errors')
    args = parser.parse_args()

    main(pooling=args.pooling, debug=args.debug)

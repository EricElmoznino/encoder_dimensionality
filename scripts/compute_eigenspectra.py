import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from activation_models.generators import get_activation_models
from custom_model_tools.eigenspectrum import ImageNetLayerEigenspectrum


def main(pooling):
    for model, layers in get_activation_models():
        if pooling:
            eigspec = ImageNetLayerEigenspectrum(model, pooling=pooling)
            eigspec.fit(layers)
        else:
            # Compute the eigenspectrum one layer at a time to avoid running out of memory
            # todo: merge pickled dictionaries together and delete the individual ones
            for layer in layers:
                identifier_append = f'|layer:{layer}'
                model.identifier += identifier_append
                eigspec = ImageNetLayerEigenspectrum(model, pooling=pooling)
                eigspec.fit([layer])
                model.identifier = model.identifier.replace(identifier_append, '')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute and store eigenspectra of models')
    parser.add_argument('--pooling', type=bool, default=True,
                        help='Whether or not to perform global max-pooling prior to computing the eigenspectrum')
    parser.add_argument('--debug', action='store_true',
                        help='Just run a single model to make sure there are no errors')
    args = parser.parse_args()

    main(pooling=args.pooling)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from activation_models.generators import counterexample_models
from custom_model_tools.eigenspectrum import EigenspectrumImageNet
from utils import timed
from scripts.fit_encoding_models import fit_encoder, get_benchmark


@timed
def main():
    benchmark = get_benchmark(benchmark='majajhong2015', region='IT', regression='pls', data_dir=None)

    eigspec_df = pd.DataFrame()
    eigmetrics_df = pd.DataFrame()
    scores_df = pd.DataFrame()

    for model, layers in counterexample_models():
        eigspec = EigenspectrumImageNet(activations_extractor=model, pooling=True)
        eigspec.fit(layers)
        eigspec_df = eigspec_df.append(eigspec.as_df())
        eigmetrics_df = eigmetrics_df.append(eigspec.metrics_as_df())
        del eigspec

        layer_scores = fit_encoder(benchmark, model, layers, True)
        scores_df = scores_df.append(layer_scores)

    eigspec_df.to_csv('results/counter-example|eigspectra.csv', index=False)
    eigmetrics_df.to_csv('results/counter-example|eigmetrics.csv', index=False)
    scores_df.to_csv('results/counter-example|encoding.csv', index=False)


if __name__ == '__main__':
    main()

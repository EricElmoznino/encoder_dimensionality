from brainscore.benchmarks._neural_common import NeuralBenchmark
from brainscore.benchmarks.majajhong2015 import load_assembly, VISUAL_DEGREES, NUMBER_OF_TRIALS, BIBTEX
from brainscore.metrics.ceiling import RDMConsistency
from brainscore.metrics.rdm import RDMCrossValidated
from brainscore.utils import LazyLoad


def _DicarloMajajHong2015Region(region, identifier_metric_suffix, similarity_metric, ceiler):
    assembly_repetition = LazyLoad(lambda region=region: load_assembly(average_repetitions=False,
                                                                       region=region,
                                                                       access='public'))
    assembly = LazyLoad(lambda region=region: load_assembly(average_repetitions=True,
                                                            region=region,
                                                            access='public'))
    return NeuralBenchmark(identifier=f'dicarlo.MajajHong2015.{region}.public-{identifier_metric_suffix}', version=3,
                           assembly=assembly, similarity_metric=similarity_metric,
                           visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
                           ceiling_func=lambda: ceiler(assembly_repetition),
                           parent=region,
                           bibtex=BIBTEX)


def DicarloMajajHong2015V4RSA():
    return _DicarloMajajHong2015Region('V4', identifier_metric_suffix='rsa',
                                       similarity_metric=RDMCrossValidated(
                                           crossvalidation_kwargs=dict(stratification_coord='object_name')),
                                       ceiler=RDMConsistency())


def DicarloMajajHong2015ITRSA():
    return _DicarloMajajHong2015Region('IT', identifier_metric_suffix='rsa',
                                       similarity_metric=RDMCrossValidated(
                                           crossvalidation_kwargs=dict(stratification_coord='object_name')),
                                       ceiler=RDMConsistency())

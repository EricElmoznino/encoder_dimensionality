import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from torchvision.models import resnet18, resnet50
from candidate_models.base_models.unsupervised_vvs import ModelBuilder
from model_tools.activations.pytorch import PytorchWrapper, load_preprocess_images
from functools import partial
from utils import properties_to_id


resnet18_pt_layers = [f'layer1.{i}.relu' for i in range(2)] + \
                     [f'layer2.{i}.relu' for i in range(2)] + \
                     [f'layer3.{i}.relu' for i in range(2)] + \
                     [f'layer4.{i}.relu' for i in range(2)]

resnet50_pt_layers = [f'layer1.{i}.relu' for i in range(3)] + \
                     [f'layer2.{i}.relu' for i in range(4)] + \
                     [f'layer3.{i}.relu' for i in range(6)] + \
                     [f'layer4.{i}.relu' for i in range(3)]

resnet18_tf_layers = [f'encode_{i}' for i in range(2, 10)]


def get_activation_models(pytorch=True, vvs=True, taskonomy=True):
    if pytorch:
        for model, layers in pytorch_models():
            yield model, layers
    if vvs:
        for model, layers in vvs_models():
            yield model, layers
    if taskonomy:
        for model, layers in taskonomy_models():
            yield model, layers


def pytorch_models():
    model = resnet18(pretrained=False)
    identifier = properties_to_id('ResNet18', 'None', 'Untrained', 'PyTorch')
    model = wrap_pt(model, identifier)
    yield model, resnet18_pt_layers

    model = resnet50(pretrained=False)
    identifier = properties_to_id('ResNet50', 'None', 'Untrained', 'PyTorch')
    model = wrap_pt(model, identifier)
    yield model, resnet50_pt_layers

    model = resnet18(pretrained=True)
    identifier = properties_to_id('ResNet18', 'Object Classification', 'Supervised', 'PyTorch')
    model = wrap_pt(model, identifier)
    yield model, resnet18_pt_layers

    model = resnet50(pretrained=True)
    identifier = properties_to_id('ResNet50', 'Object Classification', 'Supervised', 'PyTorch')
    model = wrap_pt(model, identifier)
    yield model, resnet50_pt_layers


def vvs_models():
    configs = [('resnet18-supervised', 'Object Classification', 'Supervised'),
               ('resnet18-la', 'Local Aggregation', 'Self-Supervised'),
               ('resnet18-ir', 'Instance Recognition', 'Self-Supervised'),
               ('resnet18-ae', 'Auto-Encoder', 'Self-Supervised'),
               ('resnet18-cpc', 'Contrastive Predictive Coding', 'Self-Supervised'),
               ('resnet18-color', 'Colorization', 'Self-Supervised'),
               ('resnet18-rp', 'Relative Position', 'Self-Supervised'),
               ('resnet18-depth', 'Depth Prediction', 'Supervised'),
               ('resnet18-simclr', 'SimCLR', 'Self-Supervised'),
               ('resnet18-deepcluster', 'Deep Cluster', 'Self-Supervised'),
               ('resnet18-cmc', 'Contrastive Multiview Coding', 'Self-Supervised')]

    for vvs_identifier, task, kind in configs:
        tf.reset_default_graph()

        model = ModelBuilder()(vvs_identifier)
        identifier = properties_to_id('ResNet18', task, kind, 'VVS')
        model.identifier = identifier

        if vvs_identifier in ModelBuilder.PT_MODELS:
            layers = resnet18_pt_layers
        else:
            layers = resnet18_tf_layers

        yield model, layers


def taskonomy_models():
    # todo: add taskonomy models
    return []


def wrap_pt(model, identifier, res=224):
    return PytorchWrapper(model=model,
                          preprocessing=partial(load_preprocess_images, image_size=res),
                          identifier=identifier)

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import torch
from torchvision.models import resnet18, resnet50
from candidate_models.base_models.unsupervised_vvs import ModelBuilder
from model_tools.activations.pytorch import PytorchWrapper, load_preprocess_images
from visualpriors.taskonomy_network import TASKONOMY_PRETRAINED_URLS, TaskonomyEncoder
from functools import partial
from utils import properties_to_id
from counter_example.train_cifar10 import create_cifar10_resnet18


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
    configs = [('autoencoding', 'Auto-Encoder', 'Self-Supervised'),
               ('curvature', 'Curvature Estimation', 'Supervised'),
               ('denoising', 'Denoising', 'Self-Supervised'),
               ('edge_texture', 'Edge Detection (2D)', 'Supervised'),
               ('edge_occlusion', 'Edge Detection (3D)', 'Supervised'),
               ('egomotion', 'Egomotion', 'Supervised'),
               ('fixated_pose', 'Fixated Pose Estimation', 'Supervised'),
               ('jigsaw', 'Jigsaw', 'Self-Supervised'),
               ('keypoints2d', 'Keypoint Detection (2D)', 'Supervised'),
               ('keypoints3d', 'Keypoint Detection (3D)', 'Supervised'),
               ('nonfixated_pose', 'Non-Fixated Pose Estimation', 'Supervised'),
               ('point_matching', 'Point Matching', 'Supervised'),
               ('reshading', 'Reshading', 'Supervised'),
               ('depth_zbuffer', 'Depth Estimation (Z-Buffer)', 'Supervised'),
               ('depth_euclidean', 'Depth Estimation', 'Supervised'),
               ('normal', 'Surface Normals Estimation', 'Supervised'),
               ('room_layout', 'Room Layout', 'Supervised'),
               ('segment_unsup25d', 'Unsupervised Segmentation (25D)', 'Self-Supervised'),
               ('segment_unsup2d', 'Unsupervised Segmentation (2D)', 'Self-Supervised'),
               ('segment_semantic', 'Semantic Segmentation', 'Supervised'),
               ('class_object', 'Object Classification', 'Supervised'),
               ('class_scene', 'Scene Classification', 'Supervised'),
               ('inpainting', 'Inpainting', 'Self-Supervised'),
               ('vanishing_point', 'Vanishing Point Estimation', 'Supervised')]

    for taskonomy_identifier, task, kind in configs:
        model = TaskonomyEncoder()
        model.eval()
        pretrained_url = TASKONOMY_PRETRAINED_URLS[taskonomy_identifier + '_encoder']
        checkpoint = torch.hub.load_state_dict_from_url(pretrained_url)
        model.load_state_dict(checkpoint['state_dict'])

        identifier = properties_to_id('ResNet50', task, kind, 'Taskonomy')
        model = wrap_pt(model, identifier, res=256)

        yield model, resnet50_pt_layers


def counterexample_models():
    model = create_cifar10_resnet18(pretrained_ckpt='counter_example/saved_runs/resnet18/final.ckpt')
    identifier = properties_to_id('ResNet18', 'CIFAR10', 'Supervised', 'Counter-Example')
    model = wrap_pt(model, identifier, res=32, norm=([x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                     [x / 255.0 for x in [63.0, 62.1, 66.7]]))
    yield model, resnet18_pt_layers

    model = create_cifar10_resnet18(pretrained_ckpt='counter_example/saved_runs/resnet18_scrambled_labels/final.ckpt')
    identifier = properties_to_id('ResNet18', 'CIFAR10', 'Supervised Random Labels', 'Counter-Example')
    model = wrap_pt(model, identifier, res=32, norm=([x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                     [x / 255.0 for x in [63.0, 62.1, 66.7]]))
    yield model, resnet18_pt_layers


def wrap_pt(model, identifier, res=224, norm=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))):
    preprocess = partial(load_preprocess_images, image_size=res,
                         normalize_mean=norm[0], normalize_std=norm[1])
    return PytorchWrapper(model=model,
                          preprocessing=preprocess,
                          identifier=identifier)

import numpy as np
from torchvision.models import resnet50
from torchvision import transforms
from model_tools.activations.pytorch import PytorchWrapper, load_images
from utils import properties_to_id


def get_gcl_fig5_models():
    resnet50_pt_layers = ['conv1', 'bn1', 'relu', 'maxpool'] + \
                         [f'layer1.{i}.relu' for i in range(3)] + \
                         [f'layer2.{i}.relu' for i in range(4)] + \
                         [f'layer3.{i}.relu' for i in range(6)] + \
                         [f'layer4.{i}.relu' for i in range(3)] + \
                         ['avgpool', 'fc']

    def preprocess(image_filepaths):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        images = load_images(image_filepaths)
        images = [transform(img) for img in images]
        images = np.stack(images)
        return images

    model = resnet50(pretrained=True)
    identifier = properties_to_id('ResNet50 (gcl)', 'Object Classification', 'Supervised', 'PyTorch')
    yield PytorchWrapper(model=model, preprocessing=preprocess, identifier=identifier), \
          resnet50_pt_layers

    model = resnet50(pretrained=False)
    identifier = properties_to_id('ResNet50 (gcl)', 'None', 'Untrained', 'PyTorch')
    yield PytorchWrapper(model=model, preprocessing=preprocess, identifier=identifier), \
          resnet50_pt_layers

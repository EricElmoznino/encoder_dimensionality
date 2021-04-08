import torch
from torch import nn
from torchvision.models import alexnet, resnet18, resnet50
from typing import List
from activations_models.base import Model


class AlexNet(Model):

    def __init__(self, pretrained=True, **kwargs):
        super().__init__(**kwargs)

        self.pretrained = pretrained

        torch.manual_seed(27)
        base = alexnet(pretrained=pretrained)

        conv, lin = base.features, base.classifier
        self.conv1 = nn.Sequential(*conv[0:3])
        self.conv2 = nn.Sequential(*conv[3:6])
        self.conv3 = nn.Sequential(*conv[6:8])
        self.conv4 = nn.Sequential(*conv[8:10])
        self.conv5 = nn.Sequential(*conv[10:])
        self.avgpool = base.avgpool
        self.fc6 = nn.Sequential(*lin[1:3])
        self.fc7 = nn.Sequential(*lin[4:6])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc6(x)
        x = self.fc7(x)
        return x

    def base_name(self) -> str:
        name = f'alexnet'
        if not self.pretrained:
            name += ' (untrained)'
        return name

    def model_type(self) -> str:
        return 'supervised'

    def layers(self) -> List[str]:
        return ['conv1', 'conv3', 'conv5', 'fc7']


class ResNet(Model):

    def __init__(self, kind, pretrained=True, **kwargs):
        assert kind in ['resnet18', 'resnet50']
        super().__init__(**kwargs)

        self.kind = kind
        self.block = block
        self.pretrained = pretrained

        torch.manual_seed(27)
        if kind == 'resnet18':
            self.base = resnet18(pretrained=pretrained)
        elif kind == 'resnet50':
            self.base = resnet50(pretrained=pretrained)

        self.eval()

    def extract_features(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        if self.block == 1:
            return x
        x = self.base.layer2(x)
        if self.block == 2:
            return x
        x = self.base.layer3(x)
        if self.block == 3:
            return x
        x = self.base.layer4(x)
        return x

    def base_name(self) -> str:
        name = f'{self.kind} (B={self.block})'
        if not self.pretrained:
            name = name.replace(')', ' untrained)')
        return name

    def model_type(self) -> str:
        return 'supervised'

    def input_size(self) -> Tuple[int, ...]:
        return (3, 224, 224)

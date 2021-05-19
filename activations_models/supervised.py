import torch
from torch import nn
from torchvision.models import alexnet, resnet18, resnet50
from typing import List
from .base import Model


class AlexNet(Model):

    def __init__(self, pretrained=True, **kwargs):
        super(AlexNet, self).__init__(pool_map={'conv1': 4, 'conv2': 4,
                                                'conv3': 4, 'conv4': 4,
                                                'conv5': 2},
                                      **kwargs)

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
        super(ResNet, self).__init__(pool_map={'block1': 14, 'block2': 9,
                                               'block3': 7, 'block4': 3},
                                     **kwargs)

        self.kind = kind
        self.pretrained = pretrained

        torch.manual_seed(27)
        if kind == 'resnet18':
            base = resnet18(pretrained=pretrained)
        elif kind == 'resnet50':
            base = resnet50(pretrained=pretrained)
        else:
            raise NotImplementedError()

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.block1 = base.layer1
        self.block2 = base.layer2
        self.block3 = base.layer3
        self.block4 = base.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

    def base_name(self) -> str:
        name = f'{self.kind}'
        if not self.pretrained:
            name += ' (untrained)'
        return name

    def model_type(self) -> str:
        return 'supervised'

    def layers(self) -> List[str]:
        return ['block1', 'block2', 'block3', 'block4']

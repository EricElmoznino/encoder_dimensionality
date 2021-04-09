import torch
from pl_bolts.models.self_supervised import SimCLR
from typing import List
from .base import Model


class ResNetSimCLR(Model):

    def __init__(self, kind, **kwargs):
        assert kind in ['resnet50']
        super(ResNetSimCLR, self).__init__(pool_map={'block1': 14, 'block2': 9,
                                                     'block3': 7, 'block4': 3},
                                           **kwargs)

        self.kind = kind

        torch.manual_seed(27)
        if kind == 'resnet50':
            weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/' \
                          'bolts_simclr_imagenet/simclr_imagenet.ckpt'
            simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
            base = simclr.encoder
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
        name = f'{self.kind} simclr'
        return name

    def model_type(self) -> str:
        return 'unsupervised'

    def layers(self) -> List[str]:
        return ['block1', 'block2', 'block3', 'block4']

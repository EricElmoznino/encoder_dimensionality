import torch
from torch import nn
from pl_bolts.models.self_supervised import SimCLR
from typing import Tuple
from models.base import Model


class ResNetSimCLR(Model):

    def __init__(self, kind, block, **kwargs):
        assert 1 <= block <= 4
        assert kind in ['resnet50']
        poolmap = {1: 14, 2: 9, 3: 7, 4: 3}
        super().__init__(pool=poolmap[block], **kwargs)

        self.kind = kind
        self.block = block

        torch.manual_seed(27)
        if kind == 'resnet50':
            weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
            simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
            self.base = simclr.encoder

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
        name = f'{self.kind} simclr (B={self.block})'
        return name

    def model_type(self) -> str:
        return 'unsupervised'

    def input_size(self) -> Tuple[int, ...]:
        return (3, 224, 224)

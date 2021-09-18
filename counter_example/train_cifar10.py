# Taken from a PyTorch Lightning tutorial
# https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/cifar10-baseline.html

import os
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import MultiplicativeLR
from torchmetrics.functional import accuracy

seed_everything(27)

AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)


class LitResnet(LightningModule):

    def __init__(self, lr=0.1):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_cifar10_resnet18()

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('train_loss', loss)
        self.log(f'train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9
        )
        scheduler = MultiplicativeLR(
            optimizer,
            lambda epoch: 0.95
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


def create_cifar10_resnet18(pretrained_ckpt=None):
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    if pretrained_ckpt is not None:
        state_dict = torch.load(pretrained_ckpt)['state_dict']
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    return model


def main(data_dir, scrambled_labels):
    save_dir = 'counter_example/saved_runs/resnet18'
    if scrambled_labels:
        save_dir += '_scrambled_labels'

    cifar10_dm = CIFAR10DataModule(
        data_dir=data_dir,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        normalize=True,
    )
    if scrambled_labels:
        cifar10_dm.setup()
        random.shuffle(cifar10_dm.dataset_train.dataset.targets)

    model = LitResnet()

    trainer = Trainer(
        default_root_dir=save_dir,
        checkpoint_callback=False,
        progress_bar_refresh_rate=10,
        max_epochs=90 if scrambled_labels else 30,
        gpus=AVAIL_GPUS,
        callbacks=[LearningRateMonitor(logging_interval='step')],
    )

    trainer.fit(model, datamodule=cifar10_dm)
    trainer.test(model, datamodule=cifar10_dm)

    trainer.save_checkpoint(f'{save_dir}/final.ckpt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResNet18 on CIFAR10')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Location containing CIFAR10 dataset (downloaded if does not exist)')
    parser.add_argument('--scrambled_labels', action='store_true',
                        help='Scramble training labels to overfit the data')
    args = parser.parse_args()

    main(data_dir=args.data_dir, scrambled_labels=args.scrambled_labels)

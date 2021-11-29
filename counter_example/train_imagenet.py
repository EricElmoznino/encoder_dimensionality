# Taken from a PyTorch Lightning tutorial
# https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/cifar10-baseline.html

import os
import argparse
import random

import torch
import torch.nn.functional as F
import torchvision
from pl_bolts.datamodules import ImagenetDataModule
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.optim.lr_scheduler import MultiplicativeLR
from torchmetrics.functional import accuracy

seed_everything(27)

AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 128 if AVAIL_GPUS else 32
NUM_WORKERS = int(os.cpu_count() / 2)


class LitResnet(LightningModule):

    def __init__(self, lr=0.1):
        super().__init__()

        self.save_hyperparameters()
        self.model = torchvision.models.resnet18()

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


def main(data_dir, scrambled_labels):
    save_dir = 'counter_example/saved_runs/imagenet_resnet18'
    if scrambled_labels:
        save_dir += '_scrambled_labels'
        monitor = 'train_loss'
    else:
        monitor = 'val_loss'

    imagenet_dm = ImagenetDataModule(
        data_dir=data_dir,
        meta_dir=data_dir,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    imagenet_dm.train_transforms = imagenet_dm.val_transform()
    train_dataloader, val_dataloader, test_dataloader = imagenet_dm.train_dataloader(), \
                                                        imagenet_dm.val_dataloader(), \
                                                        imagenet_dm.test_dataloader()
    if scrambled_labels:
        random.shuffle(train_dataloader.dataset.targets)
        for i in range(len(train_dataloader.dataset)):
            train_dataloader.dataset.imgs[i] = (train_dataloader.dataset.imgs[i][0],
                                                train_dataloader.dataset.targets[i])
            train_dataloader.dataset.samples[i] = (train_dataloader.dataset.samples[i][0],
                                                   train_dataloader.dataset.targets[i])

    model = LitResnet()

    trainer = Trainer(
        default_root_dir=save_dir,
        progress_bar_refresh_rate=10,
        max_epochs=90 if scrambled_labels else 30,
        gpus=AVAIL_GPUS,
        callbacks=[ModelCheckpoint(monitor=monitor, filename='best'),
                   LearningRateMonitor(logging_interval='step')],
    )

    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, test_dataloaders=test_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResNet18 on ImagetNet')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Location containing ImagetNet dataset')
    parser.add_argument('--scrambled_labels', action='store_true',
                        help='Scramble training labels to overfit the data')
    args = parser.parse_args()

    main(data_dir=args.data_dir, scrambled_labels=args.scrambled_labels)

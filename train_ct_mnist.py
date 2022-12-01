import datetime

import torch

import pytorch_lightning as pl

from ct.model.ct_mnist import CT_MNIST
from ct.data.mnist import MNIST

mnist = MNIST(batch_size=256)
model = CT_MNIST()
logger = pl.loggers.TensorBoardLogger(save_dir='logs/mnist', name=datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"), version=None)

if torch.cuda.is_available():
  print('using gpu')
  accelerator = 'gpu'
else:
  accelerator = None

trainer = pl.Trainer(max_epochs=2, val_check_interval=100, logger=logger, accelerator=accelerator, devices=1)
trainer.fit(model=model, datamodule=mnist)


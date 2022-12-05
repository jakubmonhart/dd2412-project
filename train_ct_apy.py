import datetime

import torch

import pytorch_lightning as pl

from ct.model.ct_apy import CT_aPY
from ct.data.apy import aPY

apy = aPY(batch_size=64)
model = CT_aPY()
logger = pl.loggers.TensorBoardLogger(save_dir='logs/apy', name=datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"), version=None)

if torch.cuda.is_available():
  print('using gpu')
  accelerator = 'gpu'
else:
  accelerator = None

trainer = pl.Trainer(max_epochs=200, logger=logger, accelerator=accelerator, devices=1)
trainer.fit(model=model, datamodule=apy)
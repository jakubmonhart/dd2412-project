import datetime
import argparse
import re
import os

import torch
import pytorch_lightning as pl
from ct.model.cc_apy import CC_aPY
from ct.data.apy import aPY

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
parser.add_argument("--loss_weight", action='store_true', help="Use weighted CE loss (Classes are inbalanced).")
parser.set_defaults(loss_weight=False)
parser.add_argument("--resnet", default='50', type=str, help="Resnet backbone version.")
parser.add_argument("--expl_coeff", default=0.0, type=float, help="Influence of explanation loss (concepts prediction).")

if __name__ == "__main__":
  args = parser.parse_args()

  apy = aPY(batch_size=args.batch_size)
  model = CC_aPY(args)

  if torch.cuda.is_available():
    print('using gpu')
    accelerator = 'gpu'
  else:
    accelerator = None

  trainer = pl.Trainer(max_epochs=args.epochs, logger=None, accelerator=accelerator, devices=1)
  
  trainer.fit(model=model, datamodule=apy)
import datetime
import argparse
import re
import os

import torch
import pytorch_lightning as pl
from ct.model.ct_resnet_apy import CT_ResNet_aPY
from ct.data.apy import aPY

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
parser.add_argument("--loss_weight", action='store_true', help="Use weighted CE loss (Classes are inbalanced).")
parser.set_defaults(loss_weight=False)
parser.add_argument("--resnet", default='50', type=str, help="Resnet backbone version.")
parser.add_argument("--num_heads", default=2, type=int, help="Number attention heads in CT.")
parser.add_argument("--expl_coeff", default=0.0, type=float, help="Influence of explanation loss (concepts prediction).")

if __name__ == "__main__":
  args = parser.parse_args()
  
  args.experiment_name = "{}-{}".format(
    datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items()))))
  args.save_dir = 'logs/resnet_apy'

  apy = aPY(batch_size=args.batch_size)
  model = CT_ResNet_aPY(args)

  logger = pl.loggers.TensorBoardLogger(save_dir=args.save_dir, name=args.experiment_name, version=None)
  checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_acc', mode='max', dirpath=os.path.join(args.save_dir, args.experiment_name, 'checkpoint'), verbose=True)

  if torch.cuda.is_available():
    print('using gpu')
    accelerator = 'gpu'
  else:
    accelerator = None

  trainer = pl.Trainer(max_epochs=args.epochs, logger=logger, accelerator=accelerator, devices=1, callbacks=[checkpoint_callback])
  
  trainer.fit(model=model, datamodule=apy)
  trainer.test(model=model, datamodule=apy)
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
parser.add_argument("--scheduler", default='none', type=str, help="Specify scheduler.")
parser.add_argument("--warmup_epochs", default=20, type=int, help="Number of warmup epochs for cosine scheduler.") # TODO - not mentioned in paper, but they use it for CUB dataset, might help with learning.
parser.add_argument("--loss_weight", action='store_true', help="Use weighted CE loss (Classes are inbalanced).")
parser.set_defaults(loss_weight=False)
parser.add_argument("--resnet", default='50', type=str, help="Resnet backbone version.")
parser.add_argument("--expl_coeff", default=0.0, type=float, help="Influence of explanation loss (concepts prediction).")
parser.add_argument("--note", default="", type=str, help="Experiment note.")
parser.add_argument("--dropout", default=0.0, type=float, help="Droupout rate.")

if __name__ == "__main__":
  args = parser.parse_args()

  args.experiment_name = "{}-{}".format(
    datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items()))))
  args.save_dir = 'logs/cc_apy'

  apy = aPY(batch_size=args.batch_size)
  model = CC_aPY(args)

  logger = pl.loggers.TensorBoardLogger(save_dir=args.save_dir, name=args.experiment_name, version=None)
  checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_acc', mode='max',
    dirpath=os.path.join(args.save_dir, args.experiment_name, 'checkpoint'),
    verbose=True, save_last=True)

  if torch.cuda.is_available():
    print('using gpu')
    accelerator = 'gpu'
  else:
    accelerator = None

  trainer = pl.Trainer(max_epochs=args.epochs, logger=logger, accelerator=accelerator, callbacks=[checkpoint_callback], devices=1)
  
  print("training ...")
  trainer.fit(model=model, datamodule=apy)
  
  print("testing with last model ...")
  trainer.test(model=model, datamodule=apy)

  print("testing with best model ...")
  model = CC_aPY.load_from_checkpoint(checkpoint_callback.best_model_path)
  model.test_mode = 'best'
  trainer.test(model=model, datamodule=apy)
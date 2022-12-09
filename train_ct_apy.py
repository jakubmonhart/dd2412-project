import datetime
import argparse
import re
import os

import torch
import pytorch_lightning as pl
from ct.model.ct_apy import CT_aPY
from ct.data.apy import aPY

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
parser.add_argument("--scheduler", default='none', type=str, help="Specify scheduler.")
parser.add_argument("--warmup_epochs", default=20, type=int, help="Number of warmup epochs for cosine scheduler.") # TODO - not mentioned in paper, but they use it for CUB dataset, might help with learning.
parser.add_argument("--loss_weight", action='store_true', help="Use weighted CE loss (Classes are inbalanced).")
parser.set_defaults(loss_weight=False)
parser.add_argument("--dim", default=512, type=int, help="Embedding dimnesion of CCT and CT.")
parser.add_argument("--resnet", default='50', type=str, help="Resnet backbone version.")
parser.add_argument("--cct_n_layers", default=2, type=int, help="Number of TransformerEncoderLayers in CCT.")
parser.add_argument("--cct_n_heads", default=4, type=int, help="Number attention heads in CCT.")
parser.add_argument("--cct_mlp_ratio", default=1.0, type=float, help="Sets size (relative to embedding dimension) of forward dimension of TransformerEncoderLayer in CCT.")
parser.add_argument("--num_heads", default=2, type=int, help="Number attention heads in CT.")
parser.add_argument("--expl_coeff", default=0.0, type=float, help="Influence of explanation loss (concepts prediction).")

if __name__ == "__main__":
  args = parser.parse_args()
  
  args.experiment_name = "{}-{}".format(
    datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items()))))
  args.save_dir = 'logs/ct_apy'

  apy = aPY(batch_size=args.batch_size)
  model = CT_aPY(args)

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

  trainer = pl.Trainer(max_epochs=args.epochs, logger=logger, accelerator=accelerator, devices=1, callbacks=[checkpoint_callback])

  print("training ...")
  trainer.fit(model=model, datamodule=apy)
  
  print("testing with last model ...")
  trainer.test(model=model, datamodule=apy)

  print("testing with best model ...")
  model = CT_aPY.load_from_checkpoint(checkpoint_callback.best_model_path)
  trainer.test(model=model, datamodule=apy)
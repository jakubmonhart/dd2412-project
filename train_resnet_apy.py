import datetime
import argparse
import torch
import pytorch_lightning as pl
from ct.model.ct_apy import CT_aPY
from ct.data.apy import aPY

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=80, type=int, help="Number of epochs.")
parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")

if __name__ == "__main__":
  args = parser.parse_args()
  
  path = "{}-{}".format(
    datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items()))))

  apy = aPY(batch_size=64)
  model = CT_aPY()
  logger = pl.loggers.TensorBoardLogger(save_dir='logs/resnet_apy', name=datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"), version=None)

  if torch.cuda.is_available():
    print('using gpu')
    accelerator = 'gpu'
  else:
    accelerator = None

  trainer = pl.Trainer(max_epochs=200, logger=logger, accelerator=accelerator, devices=1)
  trainer.fit(model=model, datamodule=apy)
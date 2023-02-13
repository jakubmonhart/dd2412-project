from ct.data.cub import CUB
from ct.model.cub_model import CUB_CT
import warnings
import argparse
import torch
import datetime
import pytorch_lightning as pl
import re
import os
warnings.simplefilter(action='ignore', category=FutureWarning)
from ct.model.cub_backbone import VIT_Backbone
device = 'cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
parser.add_argument("--scheduler", default='none', type=str, help="Specify scheduler.")
parser.add_argument("--warmup_epochs", default=20, type=int, help="Number of warmup epochs for cosine scheduler.") # TODO - not mentioned in paper, but they use it for CUB dataset, might help with learning.
parser.add_argument("--dim", default=1024, type=int, help="Embedding dimnesion of CCT and CT.")
parser.add_argument("--num_heads", default=16, type=int, help="Number attention heads in CT.")
parser.add_argument("--expl_coeff", default=1.0, type=float, help="Influence of explanation loss (concepts prediction).")


if __name__ == '__main__':
    args = parser.parse_args()
    args.experiment_name = "{}-{}".format(
    datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items()))))
    args.save_dir = 'logs/cub'
    dataset = CUB(args.batch_size)
    dataset.setup()
    args.n_global_concepts = dataset.n_global_attr
    args.n_spatial_concepts = dataset.n_spatial_attr
    args.n_classes = int(dataset.n_labels)
    model = CUB_CT(args)
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
    trainer.fit(model=model, datamodule=dataset)

    print("testing with last model ...")
    trainer.test(model=model, datamodule=dataset)

    print("testing with best model ...")
    model = CUB_CT.load_from_checkpoint(checkpoint_callback.best_model_path)
    model.test_mode = 'best'
    trainer.test(model=model, datamodule=dataset)


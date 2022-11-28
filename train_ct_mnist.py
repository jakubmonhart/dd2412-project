import datetime

import pytorch_lightning as pl

from ct.model.ct_mnist import CT_MNIST
from ct.data.mnist import MNIST

mnist = MNIST(batch_size=256)
mnist.setup()
model = CT_MNIST()
logger = pl.loggers.TensorBoardLogger(save_dir='logs/mnist', name=datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"), version=None)
trainer = pl.Trainer(max_epochs=2, val_check_interval=100, logger=logger)
trainer.fit(model=model, datamodule=mnist)


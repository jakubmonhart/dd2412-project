import datetime

import pytorch_lightning as pl

from ct.model.ct_apy import CT_aPY
from ct.data.apy import aPY

# apy = aPY(batch_size=2)
# model = CT_aPY()
# apy.setup()
# train = apy.train_dataloader()
# for img, (target_class, target_concept) in train:
#   break
# pred_class, attn = model.model(img)
# from ct.model.ct_apy import loss_fn
# loss = loss_fn(target_class, target_concept, pred_class, attn)
# breakpoint()

apy = aPY(batch_size=2)
model = CT_aPY()
logger = pl.loggers.TensorBoardLogger(save_dir='logs/apy', name=datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"), version=None)
trainer = pl.Trainer(max_epochs=1, val_check_interval=100, logger=logger)
trainer.fit(model=model, datamodule=apy)
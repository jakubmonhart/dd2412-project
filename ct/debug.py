from data.cub import CUB

dataset = CUB()
dataset.setup(None)

train_dl = dataset.train_dataloader()

for x,y in train_dl:
  break

breakpoint()
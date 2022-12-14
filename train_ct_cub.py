from ct.data.cub import CUB
from ct.model.cub_model import _CUB_CT
import warnings
import torch
warnings.simplefilter(action='ignore', category=FutureWarning)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    dataset = CUB()
    dataset.prepare_data()
    dataset.setup()
    model = _CUB_CT(dataset.n_global_attr, dataset.n_spatial_attr, 100, dataset.n_labels, 2)
    model.to(device)
    train_loader = dataset.train_dataloader()
    for batch in train_loader:
        batch.to(device)
        out = model(batch)
        print(batch.keys())
        break
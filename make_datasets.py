import torch
from torch_geometric.loader import DataLoader
from datasets import ABIDEDataset
from models import GATE
from tqdm import tqdm

dataset = ABIDEDataset('data/')
dataset = dataset.shuffle()

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = int(0.1 * len(dataset))

train_dataset = dataset[:train_size]
val_dataset = dataset[train_size:train_size + val_size]
test_dataset = dataset[train_size + val_size:]

torch.save(train_dataset, 'train_dataset.pt')
torch.save(val_dataset, 'val_dataset.pt')
torch.save(test_dataset, 'test_dataset.pt')



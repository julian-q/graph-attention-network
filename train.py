import torch
from torch_geometric.loader import DataLoader
from datasets import ABIDEDataset
from models import GATE
from tqdm import tqdm
import wandb
import os
import argparse
from munch import Munch

def train(model, loss_fn, device, data_loader, optimizer):
    """ Performs an epoch of model training.

    Parameters:
    model (nn.Module): Model to be trained.
    loss_fn (nn.Module): Loss function for training.
    device (torch.Device): Device used for training.
    data_loader (torch.utils.data.DataLoader): Data loader containing all batches.
    optimizer (torch.optim.Optimizer): Optimizer used to update model.

    Returns:
    float: Total loss for epoch.
    """
    model.train()
    loss = 0

    for batch in data_loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch)
        
        loss = loss_fn(out, batch.y)

        loss.backward()
        optimizer.step()

    return loss.item()

def eval(model, device, loader, num_classes):
    """ Calculate accuracy for all examples in a DataLoader.

    Parameters:
    model (nn.Module): Model to be evaluated.
    device (torch.Device): Device used for training.
    loader (torch.utils.data.DataLoader): DataLoader containing examples to test.
    """
    model.eval()

    cor = torch.zeros(num_classes)
    tot = torch.zeros(num_classes)

    for batch in loader:
        batch = batch.to(device)

        pred = torch.argmax(model(batch), 1)
        y = batch.y

        for i in range(num_classes):
            cor[i] += torch.logical_and(y == i, pred == i).sum().cpu()
            tot[i] += (y == i).sum().cpu()

    return cor / tot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    hyperparameter_defaults = {
        "hidden_channels_1": 32,
        "hidden_channels_2": 128,
        "hidden_channels_3": 32,
        "att_channels": 32,
        "dropout": 0.2,
        "heads": 4,
        "batch_size": 8,
        "learning_rate": 0.001,
        "epochs": 50
    }
    
    if args.wandb:
        wandb.init(config=hyperparameter_defaults, project="GATE-PyG", entity="julian-q")
        config = wandb.config
    else:
        config = Munch(hyperparameter_defaults)

    breakpoint()
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = torch.load('train_dataset.pt')
    val_dataset = torch.load('val_dataset.pt')
    test_dataset = torch.load('test_dataset.pt')

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    val_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)

    model = GATE(train_dataset.num_node_features, 2, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    loss_fn = torch.nn.NLLLoss()
    losses = []

    for epoch in range(config.epochs):
        loss = train(model, loss_fn, device, train_loader, optimizer)
        val_acc = eval(model, device, val_loader, 2)
        torch.save(model, 'model.pt')
        
        losses.append(loss)
        
        print(f'Epoch: {epoch + 1:02d}, '
            f'Loss: {loss:.4f}, '
            f'Val Acc: {val_acc}, ')

        wandb.log({'accuracy_0': val_acc[0].item(), 'accuracy_1': val_acc[1].item(), 'loss': loss})
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))

    torch.save(losses, 'losses.pt')

    test_acc = eval(model, device, test_loader, 2)
    print(f'Test Acc: {test_acc}')

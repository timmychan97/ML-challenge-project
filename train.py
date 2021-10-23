import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data import (TensorDataset, SequentialSampler, RandomSampler,
                                DataLoader)

from model.model import FFNetwork
import utils


def features_to_dataset(features) -> TensorDataset:
    labels = [f['label'] for f in features]
    input_ids = [f['repr'] for f in features]
    labels = torch.tensor(labels, dtype=torch.long)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    return TensorDataset(input_ids, labels)


def evaluate(model: nn.Module, features, batch_size: int, device: str) -> tuple:
    '''Return (preds, loss)'''
    dataset = features_to_dataset(features)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    loss_fn = nn.MSELoss()

    preds = []
    total_loss = 0
    total_steps = 0
    for i, batch in enumerate(dataloader):
        x, y = batch
        x.to(device)        # (B, N)
        y.to(device)        # (B,)
        pred = model(x)     # (B,) 
        total_loss += loss_fn(pred, label)
    
    loss = total_loss / (len(features) * epochs)
    return preds, total_loss
    


def train() -> None:
    # Hyperparameters
    batch_size = 32
    epochs = 6
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = nn.MSELoss()

    # Load data
    dev_features = utils.load_json_by_line('data/dev.json')
    train_features = utils.load_json_by_line('data/train.json')
    train_dataset = features_to_dataset(train_features)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    # Load model
    model = FFNetwork()

    # Optimizer

    print("*** Start training ***")
    print(f'epochs = {epochs}')
    print(f'batch size = {batch_size}')
    print(f'# train examples = {len(train_features)}')
    print(f'# dev examples = {len(dev_features)}')

    train_loss_history = []
    dev_loss_history = []

    for ep in range(epochs):
        for i, batch in enumerate(train_dataloader):
            x, y = batch
            x.to(device)
            y.to(device)

            preds = model(x)
            loss = loss_fn(preds, y)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            num_train_steps += 1

        preds, loss = evaluate(model, dev_features, batch_size, device)
        # Save model if it's best until now
        
    print('*** Training finished ***')


def main():
    train()


if __name__ == '__main__':
    main()
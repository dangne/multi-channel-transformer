import argparse
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Regressor


class BatteryDataset(Dataset):
    def __init__(self, X, y):
        # Only use current, voltage, temperature features
        self.X = X[:, :, :, :3].astype(np.float32)
        self.y = y.astype(np.float32)
        self.num_cycles, self.num_points, self.num_features = self.X[0].shape
        self.num_labels = y[0].shape[0]

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    

def evaluate(args, model, data):
    model.eval()
    avg_loss = 0
    avg_rmse = 0
    with torch.no_grad():
        for n, batch in enumerate(data):
            x, y = batch
            x = x.to(args.device)
            y = y.to(args.device)
            preds = model(x)
            loss = torch.sqrt(F.mse_loss(preds, y))
            rmse = torch.sqrt(F.mse_loss(preds, y))
            avg_loss = (loss + n*avg_loss)/(n + 1)
            avg_rmse = (rmse + n*avg_rmse)/(n + 1)
    return avg_loss, avg_rmse


def train(args, model, train_dataloader, val_dataloader, test_dataloader, optimizer, summary):
    total_steps = 0
    num_batches = len(train_dataloader)
    for epoch in range(args.num_epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            model.train()
            optimizer.zero_grad()

            x, y = batch
            x = x.to(args.device)
            y = y.to(args.device)

            preds = model(x)
            loss = torch.sqrt(F.mse_loss(preds, y))
            rmse = torch.sqrt(F.mse_loss(preds, y))

            loss.backward()
            optimizer.step()

            if (total_steps+1) % args.summary_batch_interval == 0:
                val_loss, val_rmse = evaluate(args, model, val_dataloader)
                print(f"epoch: {epoch+1}/{args.num_epochs} - batch_idx: {batch_idx+1}/{num_batches} - loss: {loss:.4f} - rmse: {rmse:.4f} - val_loss: {val_loss:.4f} - val_rmse: {val_rmse:.4f}")
                summary.add_scalars("losses", {"train_loss": loss, "val_loss": val_loss}, total_steps)
                summary.add_scalars("rmse", {"train_rmse": rmse, "val_loss": val_rmse}, total_steps)
            total_steps += 1


def main(args):
    # Define dataset
    with open(args.data_path, "rb") as f:
        dataset = pickle.load(f)
        train_data = BatteryDataset(dataset["X_train"], dataset["Y_train"]) 
        val_data = BatteryDataset(dataset["X_val"], dataset["Y_val"]) 
        test_data = BatteryDataset(dataset["X_test"], dataset["Y_test"]) 

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=args.shuffle, pin_memory=args.pin_memory)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=args.shuffle, pin_memory=args.pin_memory)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=args.shuffle, pin_memory=args.pin_memory)

    # Define model
    num_cycles, num_points, num_features = train_data.num_cycles, train_data.num_points, train_data.num_features
    num_labels = train_data.num_labels
    model = Regressor(num_labels, num_cycles, num_points, num_features, args.embed_dim, args.hidden_dim, args.dropout)
    model.to(args.device)
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    # Define summary
    summary = SummaryWriter()

    # Main training loop
    train(args, model, train_dataloader, val_dataloader, test_dataloader, optimizer, summary)
    
    test_loss, test_rmse = evaluate(args, model, test_dataloader)
    print(f"test_loss: {test_loss} - test_rmse: {test_rmse}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dataset arguments
    parser.add_argument("--data_path", type=str, default="./data/data.pkl")

    # Model arguments
    parser.add_argument("--embed_dim", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=100)

    # Training arguments
    parser.add_argument("--shuffle", default=True)
    parser.add_argument("--pin_memory", default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--summary_batch_interval", type=int, default=1)

    args = parser.parse_args()
    args.device = torch.device(args.device)
    print(args)

    main(args)
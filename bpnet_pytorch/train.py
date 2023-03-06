import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import tqdm
import os
import wandb

from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from torch.distributions.multinomial import Multinomial

from bpnet_pytorch import BPNet
from bpnet_pytorch.data import BPNetDataset
from bpnet_pytorch.util import EarlyStopping

def multinomial_nll(logit, target, reduction='batch_mean'):
    """Multinomial NLL loss.
    Note that constant terms with respect to the model prediction are ignored.
    """
    if reduction == 'batch_mean':
        return -(F.log_softmax(logit, dim=1) * target).sum() / logit.shape[0]
    elif reduction == 'sum':
        return -(F.log_softmax(logit, dim=1) * target).sum()
    elif reduction == 'none':
        return -(F.log_softmax(logit, dim=1) * target)

def train(model, train_loader, optimizer):
    model.train()

    running_profiles, running_total_counts = [], []
    running_profile_labels, running_total_count_labels = [], []

    # Training loop with progressbar.
    bar = tqdm.tqdm(train_loader, total=len(train_loader), leave=False)
    for idx, batch in enumerate(bar):
        seq = batch['seq'].cuda()
        profile = batch['profile'].cuda()
        total_count = batch['total_count'].cuda()

        optimizer.zero_grad()
        out = model(seq)

        nll = multinomial_nll(out['profile'], profile)
        total_count_mse = F.mse_loss(torch.log(1 + out['total_count']), torch.log(1 + total_count))

        loss = nll + 10 * total_count_mse  # lambda=10 from https://github.com/kundajelab/bpnet/blob/master/bpnet/premade/bpnet9-ginspec.gin#L116

        loss.backward()
        optimizer.step()

        running_profiles.append(out['profile'].detach().cpu())
        running_total_counts.append(out['total_count'].detach().cpu())
        running_profile_labels.append(profile.cpu())
        running_total_count_labels.append(total_count.cpu())

        if idx % 100 == 0:
            running_profiles = torch.cat(running_profiles, dim=0)
            running_total_counts = torch.cat(running_total_counts, dim=0)
            running_profile_labels = torch.cat(running_profile_labels, dim=0)
            running_total_count_labels = torch.cat(running_total_count_labels, dim=0)

            nll = multinomial_nll(running_profiles, running_profile_labels)
            mse = F.mse_loss(torch.log(1 + running_total_counts), torch.log(1 + running_total_count_labels))
            running_loss = nll + 10 * mse

            loss = running_loss.item()
            bar.set_postfix(loss=loss)
            wandb.log({
                'train/profile_nll': nll.detach().cpu().item(),
                'train/total_count_mse': mse.detach().cpu().item(),
                'train/loss': loss,
            })

            running_profiles, running_total_counts = [], []
            running_profile_labels, running_total_count_labels = [], []

def validate(model, val_loader):
    model.eval()

    profiles, total_counts = [], []
    profile_labels, total_count_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            seq = batch['seq'].cuda()
            profile = batch['profile'].cuda()
            total_count = batch['total_count'].cuda()

            out = model(seq)

            profiles.append(out['profile'].cpu())
            total_counts.append(out['total_count'].cpu())
            profile_labels.append(profile.cpu())
            total_count_labels.append(total_count.cpu())

    profiles = torch.cat(profiles, dim=0)
    total_counts = torch.cat(total_counts, dim=0)
    profile_labels = torch.cat(profile_labels, dim=0)
    total_count_labels = torch.cat(total_count_labels, dim=0)

    nll = multinomial_nll(profiles, profile_labels)
    total_count_mse = F.mse_loss(torch.log(1 + total_counts), torch.log(1 + total_count_labels))
    loss = nll + 10 * total_count_mse

    wandb.log({
        'val/profile_nll': nll,
        'val/total_count_mse': total_count_mse,
        'val/loss': loss,
    })

    return loss

def test(model, test_loader):
    model.eval()

    profiles, total_counts = [], []
    profile_labels, total_count_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            seq = batch['seq'].cuda()
            profile = batch['profile'].cuda()
            total_count = batch['total_count'].cuda()

            out = model(seq)

            profiles.append(out['profile'].cpu())
            total_counts.append(out['total_count'].cpu())
            profile_labels.append(profile.cpu())
            total_count_labels.append(total_count.cpu())

    profiles = torch.cat(profiles, dim=0)
    total_counts = torch.cat(total_counts, dim=0)
    profile_labels = torch.cat(profile_labels, dim=0)
    total_count_labels = torch.cat(total_count_labels, dim=0)

    nll = multinomial_nll(profiles, profile_labels)
    total_count_mse = F.mse_loss(torch.log(1 + total_counts), torch.log(1 + total_count_labels))
    loss = nll + 10 * total_count_mse

    wandb.log({
        'test/profile_nll': nll,
        'test/total_count_mse': total_count_mse,
        'test/loss': loss,
    })

    return loss

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Performance drops, so commenting out for now.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

def main():
    import pandas as pd
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--val', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--output', '-o', required=True)
    # batch size=128 from
    # https://github.com/kundajelab/bpnet/blob/1c6e5c4caf97cf34ccf716ef5b8b9c8de231cca2/bpnet/premade/bpnet9-ginspec.gin#L121
    parser.add_argument('--batch-size', type=int, default=128)

    # epochs=200 from
    # https://github.com/kundajelab/bpnet/blob/1c6e5c4caf97cf34ccf716ef5b8b9c8de231cca2/bpnet/premade/bpnet9-ginspec.gin#L78
    parser.add_argument('--epochs', type=int, default=200)

    # lr=0.004 from
    # https://github.com/kundajelab/bpnet/blob/1c6e5c4caf97cf34ccf716ef5b8b9c8de231cca2/bpnet/premade/bpnet9-ginspec.gin#L119
    parser.add_argument('--lr', type=float, default=0.004)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-wandb', action='store_true', default=False)
    args = parser.parse_args()

    seed_everything(args.seed)
    if not args.use_wandb:
        os.environ['WANDB_MODE'] = 'disabled'

    wandb.init(project='bpnet-pytorch', config=args, reinit=True)

    # Validation: Chromosomes 2, 3, 4
    # Test: Chromosomes 1, 8, 9
    # Train: Rest
    # TODO: implement dataset
    train_df = pd.read_csv(args.train)
    train_set = BPNetDataset()

    val_df = pd.read_csv(args.val)
    val_set = BPNetDataset()

    test_df = pd.read_csv(args.test)
    test_set = BPNetDataset()

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=16, pin_memory=True)

    model = BPNet()
    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # Taken from Methods section.
    early_stopping = EarlyStopping(patience=5)

    best_val_loss = np.inf
    for epoch in range(args.epochs):
        train(model, train_loader, optimizer)
        val_loss = validate(model, val_loader)
        test_loss = test(model, test_loader)

        message = f'Epoch {epoch} Validation: loss {val_loss:.4f},'
        print(message)

        message = f'Epoch {epoch} Test: loss {test_loss:.4f},'
        print(message)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.output)

        early_stopping.update(val_loss)
        if early_stopping.stop:
            break
    
    wandb.log({
        'best_val_loss': best_val_loss,
    })

if __name__ == '__main__':
    main()

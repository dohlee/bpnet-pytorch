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

def multinomial_nll(probs, target):
    """Multinomial NLL loss."""
    return -Multinomial(probs=probs).log_prob(target)

def train(model, train_loader, optimizer, criterion, metrics_f):
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
        loss = multinomial_nll(out['profile'], profile) + F.mse_loss(torch.log(1 + out['total_count']), torch.log(1 + total_count))
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

            running_loss = multinomial_nll(running_profiles, running_profile_labels) + F.mse_loss(torch.log(1 + running_total_counts), torch.log(1 + running_total_count_labels))

            loss = running_loss.item()
            bar.set_postfix(loss=loss)
            wandb.log({
                'train/loss': loss,
            })

            running_output, running_label = [], []

def validate(model, val_loader, criterion, metrics_f):
    model.eval()

    out_fwd, out_rev, label = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            wt_emb, mut_emb = batch['wt_emb'].cuda(), batch['mut_emb'].cuda()
            _label = batch['label'].cuda().flatten()

            _out_fwd = model(wt_emb, mut_emb).flatten()
            _out_rev = model(mut_emb, wt_emb).flatten()  # Swap wt_emb and mut_emb.

            out_fwd.append(_out_fwd.cpu())
            out_rev.append(_out_rev.cpu())

            label.append(_label.cpu())
        
    out_fwd = torch.cat(out_fwd, dim=0)
    out_rev = torch.cat(out_rev, dim=0)
    label = torch.cat(label, dim=0)

    loss = criterion(out_fwd, label).item()
    metrics = {k: f(out_fwd, label) for k, f in metrics_f.items()}

    # Add antisymmetry metrics.
    metrics['pearson_fr'] = pearsonr(out_fwd, out_rev)[0] 
    metrics['delta'] = torch.cat([out_fwd, out_rev], dim=0).mean()

    wandb.log({
        'val/loss': loss,
        'val/pearson': metrics['pearson'],
        'val/spearman': metrics['spearman'],
        'val/pearson_fr': metrics['pearson_fr'],
        'val/delta': metrics['delta'],
    })

    return loss, metrics

def test(model, val_loader, criterion, metrics_f):
    model.eval()

    out_fwd, out_rev, label = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            wt_emb, mut_emb = batch['wt_emb'].cuda(), batch['mut_emb'].cuda()
            _label = batch['label'].cuda().flatten()

            _out_fwd = model(wt_emb, mut_emb).flatten()
            _out_rev = model(mut_emb, wt_emb).flatten()  # Swap wt_emb and mut_emb.

            out_fwd.append(_out_fwd.cpu())
            out_rev.append(_out_rev.cpu())

            label.append(_label.cpu())
        
    out_fwd = torch.cat(out_fwd, dim=0)
    out_rev = torch.cat(out_rev, dim=0)
    label = torch.cat(label, dim=0)

    loss = criterion(out_fwd, label).item()
    metrics = {k: f(out_fwd, label) for k, f in metrics_f.items()}

    # Add antisymmetry metrics.
    metrics['pearson_fr'] = pearsonr(out_fwd, out_rev)[0] 
    metrics['delta'] = torch.cat([out_fwd, out_rev], dim=0).mean()

    wandb.log({
        'test/loss': loss,
        'test/pearson': metrics['pearson'],
        'test/spearman': metrics['spearman'],
        'test/pearson_fr': metrics['pearson_fr'],
        'test/delta': metrics['delta'],
    })

    return loss, metrics

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
    parser.add_argument('--batch-size', type=int, default=256)  # Taken from official implementation.
    parser.add_argument('--epochs', type=int, default=100)      # Taken from official implementation.
    parser.add_argument('--lr', type=float, default=0.004)      # Taken from Methods section.
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
    # TODO: Early stopping
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)
    criterion = nn.MSELoss()

    metrics_f = {
        'pearson': lambda x, y: pearsonr(x, y)[0],
        'spearman': lambda x, y: spearmanr(x, y)[0],
    }

    best_val_loss = np.inf
    best_val_pearson = -np.inf
    best_val_spearman = -np.inf
    best_test_loss = np.inf
    best_test_pearson = -np.inf
    best_test_spearman = -np.inf
    for epoch in range(args.epochs):
        train(model, train_loader, optimizer, criterion, metrics_f)
        val_loss, val_metrics = validate(model, val_loader, criterion, metrics_f)
        test_loss, test_metrics = test(model, test_loader, criterion, metrics_f)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_pearson = val_metrics['pearson']
            best_val_spearman = val_metrics['spearman']

            torch.save(model.state_dict(), args.output)

            best_test_loss = test_loss
            best_test_pearson = test_metrics['pearson']
            best_test_spearman = test_metrics['spearman']

        message = f'Epoch {epoch} Validation: loss {val_loss:.4f},'
        message += ', '.join([f'{k} {v:.4f}' for k, v in val_metrics.items()])
        print(message)

        message = f'Epoch {epoch} Test: loss {test_loss:.4f},'
        message += ', '.join([f'{k} {v:.4f}' for k, v in test_metrics.items()])
        print(message)

        scheduler.step()
    
    wandb.log({
        'best_val_loss': best_val_loss,
        'best_val_pearson': best_val_pearson,
        'best_val_spearman': best_val_spearman,
        'test_loss': best_test_loss,
        'test_pearson': best_test_pearson,
        'test_spearman': best_test_spearman,
    })

if __name__ == '__main__':
    main()

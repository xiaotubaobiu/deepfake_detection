import argparse
import os
import numpy as np
import torch

from models.classifier import DualDomainDetector
from data.df40_dataset import get_dataloader
from utils.metrics import compute_auc, compute_eer


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_scores = []

    for batch in dataloader:
        batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        logits = model(batch_gpu)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        labels = batch['label'].numpy()
        all_scores.extend(probs)
        all_labels.extend(labels)

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    auc = compute_auc(all_labels, all_scores)
    eer = compute_eer(all_labels, all_scores)
    return auc, eer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['rgb', 'freq', 'dual', 'dual_cl'])
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best_model.pth')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='outputs')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DualDomainDetector(mode=args.mode, pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'Loaded checkpoint from epoch {ckpt["epoch"]}, val AUC={ckpt["auc"]:.4f}')

    results = {}

    loader_ff = get_dataloader('test', test_domain='ff', batch_size=args.batch_size, num_workers=args.num_workers)
    auc_ff, eer_ff = evaluate(model, loader_ff, device)
    results['DF40-test-ff'] = {'auc': auc_ff, 'eer': eer_ff}
    print(f'DF40-test (ff): AUC={auc_ff:.4f}, EER={eer_ff:.4f}')

    loader_cdf = get_dataloader('test', test_domain='cdf', batch_size=args.batch_size, num_workers=args.num_workers)
    auc_cdf, eer_cdf = evaluate(model, loader_cdf, device)
    results['DF40-test-cdf'] = {'auc': auc_cdf, 'eer': eer_cdf}
    print(f'DF40-test (cdf): AUC={auc_cdf:.4f}, EER={eer_cdf:.4f}')

    exp_name = f'M4_{args.mode}' if args.mode != 'dual_cl' else 'M4_dual_cl'
    save_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'results.txt'), 'w') as f:
        f.write(f'Mode: {args.mode}\n')
        f.write(f'DF40-test-ff: AUC={auc_ff:.4f}, EER={eer_ff:.4f}\n')
        f.write(f'DF40-test-cdf: AUC={auc_cdf:.4f}, EER={eer_cdf:.4f}\n')

    print('\n' + '='*60)
    print(f'{"Model":<25} {"DF40-ff AUC":>12} {"DF40-ff EER":>12} {"DF40-cdf AUC":>12} {"DF40-cdf EER":>12}')
    print('-'*60)
    mode_label = {'rgb': 'M1: RGB-only', 'freq': 'M2: Freq-only', 'dual': 'M3: RGB+Freq', 'dual_cl': 'M4: RGB+Freq+InfoNCE'}[args.mode]
    print(f'{mode_label:<25} {auc_ff:>12.4f} {eer_ff:>12.4f} {auc_cdf:>12.4f} {eer_cdf:>12.4f}')
    print('='*60)


if __name__ == '__main__':
    main()

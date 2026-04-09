import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.classifier import DualDomainDetector
from data.df40_dataset import get_dataloader
from data.transforms import PairAugmentation
from losses.infonce import InfoNCELoss
from utils.metrics import compute_auc, compute_eer


def train_one_epoch(model, dataloader, optimizer, scaler, device, args, contrastive=False, infonce_fn=None, aug_fn=None):
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_nce_loss = 0
    n_batches = 0

    # Weighted CE: real is ~1/14 of data, give it higher weight
    class_weights = torch.tensor([13.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        if contrastive and aug_fn is not None:
            aug1_face_rgb, aug1_ctx_rgb = [], []
            aug2_face_rgb, aug2_ctx_rgb = [], []
            for i in range(batch['face_rgb'].shape[0]):
                f = (batch['face_rgb'][i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                c = (batch['context_rgb'][i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                f1, c1 = aug_fn(f, c)
                f2, c2 = aug_fn(f, c)
                aug1_face_rgb.append(torch.from_numpy(f1).permute(2, 0, 1).float() / 255.0)
                aug1_ctx_rgb.append(torch.from_numpy(c1).permute(2, 0, 1).float() / 255.0)
                aug2_face_rgb.append(torch.from_numpy(f2).permute(2, 0, 1).float() / 255.0)
                aug2_ctx_rgb.append(torch.from_numpy(c2).permute(2, 0, 1).float() / 255.0)

            batch1 = {
                'face_rgb': torch.stack(aug1_face_rgb).to(device),
                'context_rgb': torch.stack(aug1_ctx_rgb).to(device),
                'face_freq': batch['face_freq'],
                'context_freq': batch['context_freq'],
            }
            batch2 = {
                'face_rgb': torch.stack(aug2_face_rgb).to(device),
                'context_rgb': torch.stack(aug2_ctx_rgb).to(device),
                'face_freq': batch['face_freq'],
                'context_freq': batch['context_freq'],
            }
        else:
            batch1 = batch2 = batch

        optimizer.zero_grad()

        with autocast('cuda', enabled=args.amp):
            result1 = model(batch1, return_features=True)
            cls_loss = criterion(result1['logits'], batch['label'])

            if contrastive and infonce_fn is not None:
                result2 = model(batch2, return_features=True)
                nce_loss = infonce_fn(result1['projection'], result2['projection'])
                loss = cls_loss + args.lambda_nce * nce_loss
            else:
                nce_loss = torch.tensor(0.0)
                loss = cls_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_nce_loss += nce_loss.item()
        n_batches += 1

        if n_batches % 50 == 0:
            print(f'  [{n_batches}/{len(dataloader)}] loss={loss.item():.4f} cls={cls_loss.item():.4f} nce={nce_loss.item():.4f}')

    return {
        'loss': total_loss / n_batches,
        'cls_loss': total_cls_loss / n_batches,
        'nce_loss': total_nce_loss / n_batches,
    }


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_scores = []
    total_loss = 0
    n = 0
    criterion = nn.CrossEntropyLoss()

    for batch in dataloader:
        batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        logits = model(batch_gpu)
        loss = criterion(logits, batch_gpu['label'])

        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        labels = batch['label'].numpy()

        all_scores.extend(probs)
        all_labels.extend(labels)
        total_loss += loss.item()
        n += 1

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    auc = compute_auc(all_labels, all_scores)
    eer = compute_eer(all_labels, all_scores)
    avg_loss = total_loss / n

    return {'auc': auc, 'eer': eer, 'loss': avg_loss}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['rgb', 'freq', 'dual', 'dual_cl'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--tau', type=float, default=0.07)
    parser.add_argument('--lambda_nce', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--no_amp', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7', help='GPU ids to use')
    args = parser.parse_args()

    if args.no_amp:
        args.amp = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Multi-GPU setup
    gpu_ids = [int(x) for x in args.gpus.split(',')]
    n_gpus = len(gpu_ids)
    print(f'Device: cuda, Mode: {args.mode}, GPUs: {gpu_ids} ({n_gpus} cards)')

    train_loader = get_dataloader('train', batch_size=args.batch_size * n_gpus, num_workers=args.num_workers)
    val_loader = get_dataloader('val', batch_size=args.batch_size * n_gpus, num_workers=args.num_workers)

    model = DualDomainDetector(mode=args.mode, pretrained=True).to(device)
    if n_gpus > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    contrastive = args.mode == 'dual_cl'
    infonce_fn = InfoNCELoss(temperature=args.tau).to(device) if contrastive else None
    aug_fn = PairAugmentation() if contrastive else None

    scaler = GradScaler(enabled=args.amp)

    best_auc = 0
    patience_counter = 0
    exp_name = f'M4_{args.mode}' if args.mode != 'dual_cl' else 'M4_dual_cl'

    for epoch in range(args.epochs):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, args, contrastive, infonce_fn, aug_fn)
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f'Epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s) '
              f'train_loss={train_metrics["loss"]:.4f} '
              f'val_auc={val_metrics["auc"]:.4f} val_eer={val_metrics["eer"]:.4f}')

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            patience_counter = 0
            save_dir = os.path.join(args.output_dir, exp_name)
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'auc': val_metrics['auc'],
                'eer': val_metrics['eer'],
                'mode': args.mode,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f'  -> Best model saved (AUC={best_auc:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    print(f'\nTraining done. Best val AUC: {best_auc:.4f}')


if __name__ == '__main__':
    main()

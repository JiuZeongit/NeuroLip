# main.py
import argparse
import os
from os.path import dirname, join

import torch
import tqdm
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from utils.data import DVSpeakerDataset, collate_events, seed_everything
from utils.models import NeuroLipClassifier, train_step


def parse_args():
    parser = argparse.ArgumentParser("Train NeuroLip")

    # data
    parser.add_argument("--data_root", type=str, required=True, help="Path to MouthNpy root directory")
    parser.add_argument("--train_light", type=int, nargs="+", default=[1])
    parser.add_argument("--train_degree", type=int, nargs="+", default=[0])
    parser.add_argument("--test_light", type=int, nargs="+", default=[1])
    parser.add_argument("--test_degree", type=int, nargs="+", default=[45])
    parser.add_argument("--train_num", type=str, nargs="*", default=None, help="Optional digit filter for training, e.g., 0 1 2")
    parser.add_argument("--test_num", type=str, nargs="*", default=None)
    parser.add_argument("--denoise_filter_time_us", type=int, default=10000)

    # train
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--save_every_n_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_decay_gamma", type=float, default=0.5)
    parser.add_argument("--lr_decay_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)

    # model
    parser.add_argument("--num_classes", type=int, default=50)
    parser.add_argument("--voxel_bins", type=int, default=9)
    parser.add_argument("--sensor_h", type=int, default=160)
    parser.add_argument("--sensor_w", type=int, default=200)
    parser.add_argument("--crop_h", type=int, default=224)
    parser.add_argument("--crop_w", type=int, default=224)
    parser.add_argument("--compressed_channels", type=int, default=16)
    parser.add_argument("--pcr_lambda", type=float, default=0.05)
    parser.add_argument("--disable_pcr", action="store_true")

    # logging
    parser.add_argument("--log_dir", type=str, required=True)

    args = parser.parse_args()
    assert os.path.isdir(dirname(args.log_dir)) or dirname(args.log_dir) == "", f"Log directory root {dirname(args.log_dir)} not found."
    return args


def build_filter(light_list, degree_list, num_list=None):
    f = {"Light": list(light_list), "Degree": list(degree_list)}
    if num_list:
        f["Num"] = list(num_list)
    return f


def evaluate(model, loader, device, desc="Eval"):
    model.eval()
    sum_loss = 0.0
    sum_acc = 0.0

    with torch.no_grad():
        for events, labels in tqdm.tqdm(loader, desc=desc):
            events = events.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits, _ = model(events)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            acc = (logits.argmax(1) == labels).float().mean()
            sum_loss += loss.item()
            sum_acc += acc.item()

    mean_loss = sum_loss / max(len(loader), 1)
    mean_acc = sum_acc / max(len(loader), 1)
    return mean_loss, mean_acc


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)

    print(
        "----------------------------\n"
        f"Starting training\n"
        f"data_root: {args.data_root}\n"
        f"train: Light={args.train_light}, Degree={args.train_degree}\n"
        f"test:  Light={args.test_light}, Degree={args.test_degree}\n"
        f"num_epochs: {args.num_epochs}\n"
        f"batch_size: {args.batch_size}\n"
        f"device: {args.device}\n"
        f"denoise_filter_time_us: {args.denoise_filter_time_us}\n"
        f"log_dir: {args.log_dir}\n"
        "----------------------------"
    )

    train_filter = build_filter(args.train_light, args.train_degree, args.train_num)
    test_filter = build_filter(args.test_light, args.test_degree, args.test_num)

    full_train_dataset = DVSpeakerDataset(
        root_dir=args.data_root,
        train=True,
        filter_condition=train_filter,
        denoise_filter_time_us=args.denoise_filter_time_us,
    )
    test_dataset_all = DVSpeakerDataset(
        root_dir=args.data_root,
        train=False,
        filter_condition=test_filter,
        denoise_filter_time_us=args.denoise_filter_time_us,
    )

    num_samples = len(full_train_dataset)
    train_num = int(num_samples * 0.8)
    perm = torch.randperm(num_samples)
    train_indices = perm[:train_num]
    val_indices = perm[train_num:]

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)
    test_dataset = Subset(test_dataset_all, torch.arange(len(test_dataset_all)))

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    loader_kwargs = dict(
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        pin_memory=args.pin_memory,
        collate_fn=collate_events,
    )
    training_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = NeuroLipClassifier(
        voxel_dimension=(args.voxel_bins, args.sensor_h, args.sensor_w),
        crop_dimension=(args.crop_h, args.crop_w),
        compressed_channels=args.compressed_channels,
        num_classes=args.num_classes,
        pretrained=False,
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay_gamma)

    writer = SummaryWriter(args.log_dir)
    iteration = 0
    best_val_loss = float("inf")

    for epoch in range(args.num_epochs):
        # Validation
        print(f"\nValidation [{epoch + 1:03d}/{args.num_epochs:03d}]")
        val_loss, val_acc = evaluate(model, validation_loader, args.device, desc="Validation")
        print(f"Validation Loss {val_loss:.4f}  Accuracy {val_acc:.4f}")
        writer.add_scalar("validation/loss", val_loss, iteration)
        writer.add_scalar("validation/accuracy", val_acc, iteration)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "iteration": iteration,
                    "args": vars(args),
                },
                join(args.log_dir, "model_best.pth"),
            )
            print(f"New best checkpoint: val_loss={best_val_loss:.4f}")

        if (epoch % args.save_every_n_epochs) == 0:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "iteration": iteration,
                    "args": vars(args),
                },
                join(args.log_dir, f"checkpoint_{iteration:05d}_{best_val_loss:.4f}.pth"),
            )

        # Training
        model.train()
        sum_loss = 0.0
        sum_acc = 0.0
        print(f"Training   [{epoch + 1:03d}/{args.num_epochs:03d}]")
        for events, labels in tqdm.tqdm(training_loader, desc="Training"):
            events = events.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)

            loss_val, pred_labels = train_step(
                model,
                events,
                labels,
                optimizer,
                use_pcr=(not args.disable_pcr),
                pcr_lambda=args.pcr_lambda,
            )
            acc = (pred_labels == labels).float().mean().item()

            sum_loss += loss_val
            sum_acc += acc
            iteration += 1

        if (epoch + 1) % args.lr_decay_every == 0:
            lr_scheduler.step()

        train_loss = sum_loss / max(len(training_loader), 1)
        train_acc = sum_acc / max(len(training_loader), 1)
        print(f"Training Iteration {iteration:6d}  Loss {train_loss:.4f}  Accuracy {train_acc:.4f}")
        writer.add_scalar("training/loss", train_loss, iteration)
        writer.add_scalar("training/accuracy", train_acc, iteration)

    # Final test using best checkpoint
    print("\nLoading best checkpoint for final test...")
    best_ckpt = torch.load(join(args.log_dir, "model_best.pth"), map_location=args.device)
    model.load_state_dict(best_ckpt["state_dict"])
    model.to(args.device)

    print("Testing...")
    test_loss, test_acc = evaluate(model, test_loader, args.device, desc="Test")
    print(f"Test Loss {test_loss:.4f}  Test Accuracy {test_acc:.4f}")
    writer.add_scalar("test/loss", test_loss, iteration)
    writer.add_scalar("test/accuracy", test_acc, iteration)
    writer.close()


if __name__ == "__main__":
    main()

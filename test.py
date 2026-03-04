# test.py
import argparse

import torch
import tqdm
from torch.utils.data import DataLoader

from utils.data import DVSpeakerDataset, collate_events, seed_everything
from utils.models import NeuroLipClassifier


def parse_args():
    parser = argparse.ArgumentParser("Evaluate NeuroLip checkpoint")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--denoise_filter_time_us", type=int, default=10000)

    # test filter
    parser.add_argument("--test_light", type=int, nargs="+", default=[1])
    parser.add_argument("--test_degree", type=int, nargs="+", default=[45])
    parser.add_argument("--test_num", type=str, nargs="*", default=None)

    # fallback model shape args (used if checkpoint lacks args)
    parser.add_argument("--num_classes", type=int, default=50)
    parser.add_argument("--voxel_bins", type=int, default=9)
    parser.add_argument("--sensor_h", type=int, default=160)
    parser.add_argument("--sensor_w", type=int, default=200)
    parser.add_argument("--crop_h", type=int, default=224)
    parser.add_argument("--crop_w", type=int, default=224)
    parser.add_argument("--compressed_channels", type=int, default=16)
    return parser.parse_args()


def build_filter(light_list, degree_list, num_list=None):
    f = {"Light": list(light_list), "Degree": list(degree_list)}
    if num_list:
        f["Num"] = list(num_list)
    return f


def evaluate(model, loader, device):
    model.eval()
    sum_loss = 0.0
    sum_acc = 0.0
    with torch.no_grad():
        for events, labels in tqdm.tqdm(loader, desc="Test"):
            events = events.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits, _ = model(events)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            acc = (logits.argmax(1) == labels).float().mean()
            sum_loss += loss.item()
            sum_acc += acc.item()
    return sum_loss / max(len(loader), 1), sum_acc / max(len(loader), 1)


def main():
    args = parse_args()
    seed_everything(args.seed)

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    model = NeuroLipClassifier(
        voxel_dimension=(ckpt_args.get("voxel_bins", args.voxel_bins), ckpt_args.get("sensor_h", args.sensor_h), ckpt_args.get("sensor_w", args.sensor_w)),
        crop_dimension=(ckpt_args.get("crop_h", args.crop_h), ckpt_args.get("crop_w", args.crop_w)),
        compressed_channels=ckpt_args.get("compressed_channels", args.compressed_channels),
        num_classes=ckpt_args.get("num_classes", args.num_classes),
        pretrained=False,
    ).to(args.device)
    model.load_state_dict(ckpt["state_dict"])

    test_filter = build_filter(args.test_light, args.test_degree, args.test_num)
    test_dataset = DVSpeakerDataset(
        root_dir=args.data_root,
        train=False,
        filter_condition=test_filter,
        denoise_filter_time_us=args.denoise_filter_time_us,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_events,
    )

    print("----------------------------")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test filter: Light={args.test_light}, Degree={args.test_degree}, Num={args.test_num}")
    print(f"Samples: {len(test_dataset)}")
    print(f"Device: {args.device}")
    print(f"denoise_filter_time_us: {args.denoise_filter_time_us}")
    print("----------------------------")

    test_loss, test_acc = evaluate(model, test_loader, args.device)
    print(f"Test Loss {test_loss:.4f}  Test Accuracy {test_acc:.4f}")


if __name__ == "__main__":
    main()

import os
import argparse

import torch
import torchvision

from .utils import PennFudanDataset, collate_fn


def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PennFudanDataset(args.dataset, train=True)
    dataset_test = PennFudanDataset(args.dataset, train=False)

    indices = torch.randperm(len(dataset)).tolist()
    split = int(len(indices) * 0.8)
    dataset = torch.utils.data.Subset(dataset, indices[:split])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[split:])

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    num_classes = 2
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    for epoch in range(args.epochs):
        train_one_epoch(model, optimizer, data_loader, device)
        torch.save(model.state_dict(), os.path.join(args.output, f"model_{epoch}.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Mask R-CNN on PennFudanPed")
    parser.add_argument("dataset", help="Path to PennFudanPed root directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--output", default=".", help="Directory to save checkpoints")
    args = parser.parse_args()
    main(args)

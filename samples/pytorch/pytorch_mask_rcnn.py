"""PyTorch Mask R-CNN Example

This sample demonstrates running inference with torchvision's
`maskrcnn_resnet50_fpn` model. It loads a pre-trained model,
performs instance segmentation on a given image, and displays the
results using Matplotlib.
"""

import argparse
from PIL import Image
import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np


def run_inference(image_path, threshold=0.7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    model.eval()

    img = Image.open(image_path).convert("RGB")
    transform = T.ToTensor()
    input_tensor = transform(img).to(device)

    with torch.no_grad():
        outputs = model([input_tensor])[0]

    boxes = outputs['boxes']
    scores = outputs['scores']
    masks = outputs['masks']

    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img)
    for box, score, mask in zip(boxes, scores, masks):
        if score < threshold:
            continue
        x1, y1, x2, y2 = box.int().tolist()
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                              linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        m = mask[0].cpu().numpy()
        ax.imshow(np.ma.masked_where(m < 0.5, m), alpha=0.5)

    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Mask R-CNN Example")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Score threshold for displaying instances")
    args = parser.parse_args()
    run_inference(args.image, args.threshold)

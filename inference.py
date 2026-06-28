"""
inference.py — End-to-end two-stage inference for Soybean Disease Classification.

Usage:
    python inference.py --image path/to/image.jpg \
                        --stage1_weights leaf_classifier_resnet18.pth \
                        --stage2_weights best_mobilenet_soybean_model_finetuned.pth

Requirements:
    pip install torch torchvision pillow
"""

import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ── Constants ─────────────────────────────────────────────────────────────────
DISEASE_CLASSES = ["Healthy", "Yellow Mosaic", "Rust", "Sudden Death Syndrome"]
LEAF_CLASSES    = ["leaf", "non_leaf"]   # must match the order the Stage 1 model was trained with

# ImageNet normalisation — same as training
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Model loaders ─────────────────────────────────────────────────────────────
def load_stage1_model(weights_path: str, device: torch.device) -> nn.Module:
    """Load the ResNet18 leaf/non-leaf binary classifier."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model


def load_stage2_model(weights_path: str, device: torch.device) -> nn.Module:
    """Load the MobileNetV2 disease classifier."""
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 4)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(image_path: str, stage1_model: nn.Module,
            stage2_model: nn.Module, device: torch.device) -> dict:
    """
    Run the full two-stage pipeline on a single image.

    Returns a dict with keys:
        - is_leaf     (bool)
        - disease     (str | None)  — None if image is rejected at Stage 1
        - confidence  (float | None)
    """
    img = Image.open(image_path).convert("RGB")
    tensor = INFERENCE_TRANSFORM(img).unsqueeze(0).to(device)

    # Stage 1 — Leaf / Non-Leaf
    with torch.no_grad():
        out1 = stage1_model(tensor)
        prob1 = torch.softmax(out1, dim=1)
        pred1 = prob1.argmax(dim=1).item()

    if LEAF_CLASSES[pred1] != "leaf":
        return {"is_leaf": False, "disease": None, "confidence": None}

    # Stage 2 — Disease Classification
    with torch.no_grad():
        out2 = stage2_model(tensor)
        prob2 = torch.softmax(out2, dim=1)
        conf, pred2 = prob2.max(dim=1)

    return {
        "is_leaf": True,
        "disease": DISEASE_CLASSES[pred2.item()],
        "confidence": round(conf.item() * 100, 2),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Soybean Disease Classification — two-stage inference"
    )
    parser.add_argument("--image",          required=True,  help="Path to input image")
    parser.add_argument("--stage1_weights", required=True,  help="Path to Stage 1 ResNet18 .pth file")
    parser.add_argument("--stage2_weights", required=True,  help="Path to Stage 2 MobileNetV2 .pth file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading Stage 1 model (ResNet18 leaf detector)...")
    stage1 = load_stage1_model(args.stage1_weights, device)

    print("Loading Stage 2 model (MobileNetV2 disease classifier)...")
    stage2 = load_stage2_model(args.stage2_weights, device)

    print(f"\nRunning inference on: {args.image}")
    result = predict(args.image, stage1, stage2, device)

    print("\n── Result ───────────────────────────────────")
    if not result["is_leaf"]:
        print("❌  REJECTED — Image does not contain a soybean leaf.")
    else:
        print(f"✅  Leaf detected.")
        print(f"🌿  Predicted disease : {result['disease']}")
        print(f"📊  Confidence        : {result['confidence']}%")
    print("─────────────────────────────────────────────")


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "models\\skin_model_1.pth"
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
NUM_CLASSES = len(CLASS_NAMES)

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def detect_architecture(state_dict):
    keys = list(state_dict.keys())
    if "conv_stem.weight" in keys:
        return "timm_efficientnet"
    if any("features" in k for k in keys):
        return "torchvision_efficientnet"
    return "resnet50"

def build_model(arch):
    if arch == "timm_efficientnet":
        import timm
        model = timm.create_model("efficientnet_b1", num_classes=NUM_CLASSES, pretrained=False)
    elif arch == "torchvision_efficientnet":
        model = models.efficientnet_b1(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    else:
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

def load_model(device):
    state = torch.load(MODEL_PATH, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    arch = detect_architecture(state)
    print(f"[INFO] Model 1 detected architecture: {arch}")

    if arch == "timm_efficientnet":
        state = {
            k.replace("classifier.1.", "classifier.") if k.startswith("classifier.1.") else k: v
            for k, v in state.items()
        }
    if arch == "resnet50":
        state = {
            k.replace("fc.1.", "fc.") if k.startswith("fc.1.") else k: v
            for k, v in state.items()
        }

    model = build_model(arch)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def predict(image_path, model, device):
    img = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze()
    top_idx = probs.argmax().item()
    return {
        "predicted_class": CLASS_NAMES[top_idx],
        "confidence": round(probs[top_idx].item(), 4),
        "all_probabilities": {
            CLASS_NAMES[i]: round(probs[i].item(), 4) for i in range(NUM_CLASSES)
        },
    }
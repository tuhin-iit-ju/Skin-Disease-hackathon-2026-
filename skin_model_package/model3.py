import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm

MODEL_PATH = os.path.join("models", "skin_model_3.pth")

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
NUM_CLASSES = len(CLASS_NAMES)

TRANSFORM = transforms.Compose([
    transforms.Resize((300, 300)),   # EfficientNet-B3 input
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def load_model(device):

    model = timm.create_model(
        "efficientnet_b3",
        pretrained=False,
        num_classes=NUM_CLASSES
    )

    state = torch.load(MODEL_PATH, map_location=device)

    if "state_dict" in state:
        state = state["state_dict"]

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
        "confidence": probs[top_idx].item(),
        "all_probabilities": {
            CLASS_NAMES[i]: probs[i].item()
            for i in range(NUM_CLASSES)
        },
    }
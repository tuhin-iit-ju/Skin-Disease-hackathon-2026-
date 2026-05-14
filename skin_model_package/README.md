# Skin Lesion Classifier — Dual Model Package

Classifies a skin lesion image into one of **7 categories** using two
fine-tuned ResNet50 models. Both models run on every image and the one
with the **higher confidence score** is returned as the final answer.

---

## 📁 Folder Structure

```
skin_model_package/
├── model1.py          ← loads & runs skin_model_1.pth
├── model2.py          ← loads & runs skin_model_f2.pth
├── predict.py         ← main script: runs both, picks the winner
├── requirements.txt   ← Python dependencies
└── README.md
```

> ⚠️ Place your two `.pth` files in the **same folder** as these scripts:
> - `skin_model_1.pth`
> - `skin_model_f2.pth`

---

## 🏷️ Classes

| Short Label | Full Name |
|-------------|-----------|
| `akiec` | Actinic Keratoses / Intraepithelial Carcinoma |
| `bcc`   | Basal Cell Carcinoma |
| `bkl`   | Benign Keratosis-like Lesions |
| `df`    | Dermatofibroma |
| `mel`   | Melanoma |
| `nv`    | Melanocytic Nevi |
| `vasc`  | Vascular Lesions |

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Command line
```bash
python predict.py path/to/skin_image.jpg
```

### In your own Python code
```python
from predict import run

result = run("path/to/skin_image.jpg")

print(result["winner"])           # e.g. "Model 2 (skin_model_f2.pth)"
print(result["predicted_class"])  # e.g. "mel"
print(result["description"])      # e.g. "Melanoma"
print(result["confidence"])       # e.g. 0.9231
```

### Full result dictionary
```python
{
    "winner":          "Model 2 (skin_model_f2.pth)",
    "predicted_class": "mel",
    "description":     "Melanoma",
    "confidence":      0.9231,
    "model1_result": {
        "predicted_class": "mel",
        "confidence": 0.8810,
        "all_probabilities": { "akiec": 0.012, "bcc": 0.005, ... }
    },
    "model2_result": {
        "predicted_class": "mel",
        "confidence": 0.9231,
        "all_probabilities": { "akiec": 0.008, "bcc": 0.003, ... }
    }
}
```

---

## 🖥️ Requirements

- Python 3.8+
- PyTorch (CPU or GPU)
- Input image: any common format (JPG, PNG, BMP, etc.)
- GPU is used automatically if available; falls back to CPU otherwise.

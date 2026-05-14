import os
import torch
import model1
import model2

CLASS_INFO = {
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis-like Lesions",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi",
    "vasc":  "Vascular Lesions",
}


def run(image_path, device):
    m1 = model1.load_model(device)
    m2 = model2.load_model(device)

    r1 = model1.predict(image_path, m1, device)
    r2 = model2.predict(image_path, m2, device)

    if r1["confidence"] >= r2["confidence"]:
        winner, best = "Model 1 (skin_model_1.pth)", r1
    else:
        winner, best = "Model 2 (skin_model_f2.pth)", r2

    return {
        "winner":          winner,
        "predicted_class": best["predicted_class"],
        "description":     CLASS_INFO.get(best["predicted_class"], "Unknown"),
        "confidence":      best["confidence"],
        "model1_result":   r1,
        "model2_result":   r2,
    }


def print_report(result):
    sep = "─" * 52
    print(f"\n{sep}")
    print("  SKIN LESION CLASSIFICATION RESULT")
    print(sep)
    print(f"  Best Model  : {result['winner']}")
    print(f"  Prediction  : {result['predicted_class'].upper()} — {result['description']}")
    print(f"  Confidence  : {result['confidence'] * 100:.2f}%")
    print(sep)

    for label, res in [("Model 1", result["model1_result"]), ("Model 2", result["model2_result"])]:
        print(f"\n  ── {label} ──")
        print(f"  Prediction : {res['predicted_class']}  ({res['confidence']*100:.2f}%)")
        for cls, prob in res["all_probabilities"].items():
            bar = "█" * int(prob * 30)
            print(f"    {cls:6s}  {prob*100:5.2f}%  {bar}")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 52)
    print("   SKIN LESION CLASSIFIER — Dual Model")
    print("=" * 52)

    while True:
        image_path = input("\nEnter image path (or 'q' to quit): ").strip().strip('"').strip("'")

        if image_path.lower() == 'q':
            print("Goodbye!")
            break

        if not os.path.exists(image_path):
            print(f"[ERROR] File not found: {image_path}")
            continue

        print_report(run(image_path, device))

        if input("Test another image? (y/n): ").strip().lower() != 'y':
            print("Goodbye!")
            break
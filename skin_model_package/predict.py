import os
import torch

import model1
import model2
import model3

from collections import defaultdict

CLASS_INFO = {
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis-like Lesions",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi",
    "vasc":  "Vascular Lesions",
}


def ensemble_predict(results):

    votes = defaultdict(int)
    confidence_sum = defaultdict(float)

    for r in results:

        cls = r["predicted_class"].strip().lower()

        votes[cls] += 1
        confidence_sum[cls] += r["confidence"]

    best_class = sorted(
        votes.keys(),
        key=lambda c: (votes[c], confidence_sum[c]),
        reverse=True
    )[0]

    return {
        "predicted_class": best_class,
        "full_name": CLASS_INFO.get(best_class, "Unknown Disease"),
        "confidence": confidence_sum[best_class] / votes[best_class],
        "votes": votes[best_class]
    }


def run(image_path, device, m1, m2, m3):

    r1 = model1.predict(image_path, m1, device)
    r2 = model2.predict(image_path, m2, device)
    r3 = model3.predict(image_path, m3, device)

    final_result = ensemble_predict([r1, r2, r3])

    return {
        "final_result": final_result,
        "model1": r1,
        "model2": r2,
        "model3": r3,
    }


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("         SKIN LESION ENSEMBLE CLASSIFIER")
    print("=" * 60)

    print("\nLoading models...")

    m1 = model1.load_model(device)
    m2 = model2.load_model(device)
    m3 = model3.load_model(device)

    print("All models loaded successfully.")

    while True:

        image_path = input(
            "\nEnter image path (or 'q' to quit): "
        ).strip()

        if image_path.lower() == 'q':
            print("Goodbye!")
            break

        if not os.path.exists(image_path):
            print("[ERROR] File not found.")
            continue

        try:

            result = run(image_path, device, m1, m2, m3)

            final = result["final_result"]

            print("\n" + "─" * 60)
            print("                 FINAL PREDICTION")
            print("─" * 60)

            print(
                f"Predicted Class : "
                f"{final['predicted_class'].upper()}"
            )

            print(
                f"Disease Name    : "
                f"{final['full_name']}"
            )

            print(
                f"Confidence      : "
                f"{final['confidence'] * 100:.2f}%"
            )

            print(
                f"Votes           : "
                f"{final['votes']} / 3"
            )

            print("\n--- Individual Model Predictions ---")

            for i, r in enumerate(
                [result["model1"],
                 result["model2"],
                 result["model3"]],
                start=1
            ):

                cls = r["predicted_class"].strip().lower()

                full_name = CLASS_INFO.get(
                    cls,
                    "Unknown Disease"
                )

                print(f"\nModel {i}")

                print(
                    f"Prediction : "
                    f"{cls.upper()}"
                )

                print(
                    f"Disease    : "
                    f"{full_name}"
                )

                print(
                    f"Confidence : "
                    f"{r['confidence'] * 100:.2f}%"
                )

            print("─" * 60)

        except Exception as e:
            print(f"[ERROR] {e}")

        again = input(
            "\nTest another image? (y/n): "
        ).strip().lower()

        if again != 'y':
            print("Goodbye!")
            break
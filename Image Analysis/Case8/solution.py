import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT_IMAGE = ROOT / "Sample_Image.png"
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)
FILTERS = ("Mean", "Gaussian", "Median", "Bilateral")

image_bgr = cv2.imread(str(INPUT_IMAGE), cv2.IMREAD_COLOR)
if image_bgr is None:
    raise FileNotFoundError(f"Could not read input image: {INPUT_IMAGE}")
clean = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
rng = np.random.default_rng(2026)
gaussian_noisy = np.clip(
    clean.astype(np.float32) + rng.normal(0.0, 18.0, clean.shape), 0, 255
).astype(np.uint8)
impulse_noisy = clean.copy()
mask = rng.random(clean.shape[:2])
impulse_noisy[mask < 0.02] = 0
impulse_noisy[(mask >= 0.02) & (mask < 0.04)] = 255


def filter_image(image: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "Mean": cv2.blur(image, (5, 5)),
        "Gaussian": cv2.GaussianBlur(image, (5, 5), 1.2),
        "Median": cv2.medianBlur(image, 5),
        "Bilateral": cv2.bilateralFilter(image, 9, 60, 60),
    }


def mse_psnr(reference: np.ndarray, result: np.ndarray) -> tuple[float, float]:
    error = float(np.mean(
        (reference.astype(np.float64) - result.astype(np.float64)) ** 2
    ))
    psnr = float("inf") if error == 0 else 10 * np.log10((255.0 ** 2) / error)
    return error, float(psnr)


def save_rgb(name: str, image: np.ndarray) -> None:
    path = OUTPUT / name
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not save image: {path}")


save_rgb("clean_original.png", clean)
save_rgb("gaussian_noisy.png", gaussian_noisy)
save_rgb("salt_pepper_noisy.png", impulse_noisy)

rows = []
panels = [
    ("Clean reference", clean),
    ("Gaussian noise", gaussian_noisy),
    ("Salt-and-pepper noise", impulse_noisy),
]
for noise_name, noisy in (
    ("Gaussian", gaussian_noisy), ("Salt-and-pepper", impulse_noisy)
):
    for filter_name, result in filter_image(noisy).items():
        mse, psnr = mse_psnr(clean, result)
        rows.append({
            "Noise": noise_name,
            "Filter": filter_name,
            "MSE": mse,
            "PSNR_dB": psnr,
        })
        save_rgb(f"{noise_name.lower().replace('-', '_')}_{filter_name.lower()}.png",
                 result)
        panels.append((f"{noise_name}: {filter_name}", result))

with (OUTPUT / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=("Noise", "Filter", "MSE", "PSNR_dB"))
    writer.writeheader()
    writer.writerows(rows)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
x = np.arange(len(FILTERS))
for ax, noise_name in zip(axes, ("Gaussian", "Salt-and-pepper")):
    values = [
        next(row["PSNR_dB"] for row in rows
             if row["Noise"] == noise_name and row["Filter"] == name)
        for name in FILTERS
    ]
    ax.bar(x, values)
    ax.set_xticks(x, FILTERS, rotation=20)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(noise_name + " noise")
    ax.grid(axis="y", alpha=0.25)
fig.suptitle("Case 8: Denoising comparison (higher PSNR is better)")
fig.tight_layout()
fig.savefig(OUTPUT / "psnr_comparison.png", dpi=160, bbox_inches="tight")
plt.close(fig)

columns = 3
rows_count = (len(panels) + columns - 1) // columns
fig, axes = plt.subplots(rows_count, columns,
                         figsize=(5 * columns, 4 * rows_count), squeeze=False)
for ax, (title, result) in zip(axes.flat, panels):
    ax.imshow(result)
    ax.set_title(title)
    ax.axis("off")
for ax in axes.flat[len(panels):]:
    ax.axis("off")
fig.suptitle("Case 8: Visual comparison of spatial filters")
fig.tight_layout()
fig.savefig(OUTPUT / "filter_comparison.png", dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"Case 8 results saved under: {OUTPUT}")

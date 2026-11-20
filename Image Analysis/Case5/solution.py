from pathlib import Path

import cv2
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
INPUT_IMAGE = ROOT / "Sample_Image.png"
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)

image_bgr = cv2.imread(str(INPUT_IMAGE), cv2.IMREAD_COLOR)
if image_bgr is None:
    raise FileNotFoundError(f"Could not read input image: {INPUT_IMAGE}")
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
smoothed = cv2.GaussianBlur(gray, (5, 5), 1.0)

sobel_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
sobel = cv2.convertScaleAbs(cv2.magnitude(sobel_x, sobel_y))
laplacian = cv2.convertScaleAbs(
    cv2.Laplacian(smoothed, cv2.CV_32F, ksize=3)
)

for name, result in (
    ("original.png", image), ("grayscale_smoothed.png", smoothed),
    ("sobel_edges.png", sobel), ("laplacian_edges.png", laplacian),
):
    saved = cv2.cvtColor(result, cv2.COLOR_RGB2BGR) if result.ndim == 3 else result
    if not cv2.imwrite(str(OUTPUT / name), saved):
        raise OSError(f"Could not save image: {OUTPUT / name}")

panels = [
    ("Leaf / foliage sample", image), ("Smoothed grayscale", smoothed),
    ("Sobel magnitude", sobel), ("Laplacian response", laplacian),
]
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for ax, (title, result) in zip(axes.flat, panels):
    ax.imshow(result, cmap="gray" if result.ndim == 2 else None,
              vmin=0 if result.ndim == 2 else None,
              vmax=255 if result.ndim == 2 else None)
    ax.set_title(title)
    ax.axis("off")
fig.suptitle("Case 5: Leaf and foliage edge detection")
fig.tight_layout()
fig.savefig(OUTPUT / "comparison.png", dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"Case 5 results saved under: {OUTPUT}")

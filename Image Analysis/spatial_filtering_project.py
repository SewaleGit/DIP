"""Spatial image filtering mini-project: runnable solutions for all eight cases.

Run ``python spatial_filtering_project.py`` to create the original-image copies,
processed images, comparison figures, and Case 8 metric table under ``outputs``.
Optional replacement images can be placed in ``inputs`` using the filenames
listed in README.md. Public sample images from scikit-image are used otherwise.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import data


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
ORIGINALS = OUTPUT / "originals"
RESULTS = OUTPUT / "results"
FIGURES = OUTPUT / "figures"
RNG = np.random.default_rng(2026)

FILTER_NAMES = ("Mean", "Gaussian", "Median", "Bilateral")


def as_uint8(image: np.ndarray) -> np.ndarray:
    """Convert sample data or loaded images to uint8 without changing uint8 data."""
    if image.dtype == np.uint8:
        return image
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        if image.size and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(image, 0, 255).astype(np.uint8)


def load_image(key: str, sample_factory, input_dir: Path) -> np.ndarray:
    """Load an optional user image or use a public scikit-image sample."""
    for extension in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        candidate = input_dir / f"{key}{extension}"
        if candidate.is_file():
            image = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read input image: {candidate}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return as_uint8(sample_factory())


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = as_uint8(image)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save image: {path}")


def gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def plot_grid(title: str, panels: list[tuple[str, np.ndarray]], path: Path,
              columns: int = 3, cmap: str | None = None) -> None:
    rows = (len(panels) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows),
                             squeeze=False)
    for ax, (label, image) in zip(axes.flat, panels):
        ax.imshow(image, cmap=cmap if image.ndim == 2 else None,
                  vmin=0 if image.ndim == 2 else None,
                  vmax=255 if image.ndim == 2 else None)
        ax.set_title(label)
        ax.axis("off")
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def add_gaussian_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    noisy = image.astype(np.float32) + RNG.normal(0.0, sigma, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image: np.ndarray, amount: float) -> np.ndarray:
    noisy = image.copy()
    mask = RNG.random(image.shape[:2])
    noisy[mask < amount / 2] = 0
    noisy[(mask >= amount / 2) & (mask < amount)] = 255
    return noisy


def laplacian_sharpen(image: np.ndarray) -> np.ndarray:
    lap = cv2.Laplacian(image, cv2.CV_32F, ksize=3)
    return np.clip(image.astype(np.float32) - 0.7 * lap, 0, 255).astype(np.uint8)


def unsharp_mask(image: np.ndarray, amount: float = 1.5) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def sobel_edges(image: np.ndarray) -> np.ndarray:
    dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.convertScaleAbs(cv2.magnitude(dx, dy))


def prewitt_edges(image: np.ndarray) -> np.ndarray:
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], np.float32)
    dx = cv2.filter2D(image, cv2.CV_32F, kernel_x)
    dy = cv2.filter2D(image, cv2.CV_32F, kernel_y)
    return cv2.convertScaleAbs(cv2.magnitude(dx, dy))


def spatial_filters(image: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "Mean": cv2.blur(image, (5, 5)),
        "Gaussian": cv2.GaussianBlur(image, (5, 5), 1.2),
        "Median": cv2.medianBlur(image, 5),
        "Bilateral": cv2.bilateralFilter(image, 9, 60, 60),
    }


def mse_psnr(reference: np.ndarray, result: np.ndarray) -> tuple[float, float]:
    error = np.mean((reference.astype(np.float64) - result.astype(np.float64)) ** 2)
    psnr = float("inf") if error == 0 else 10 * np.log10((255.0 ** 2) / error)
    return float(error), float(psnr)


def run_case_1(input_dir: Path) -> None:
    image = load_image("case1_document", data.page, input_dir)
    grayscale = gray(image)
    laplacian = laplacian_sharpen(grayscale)
    unsharp = unsharp_mask(grayscale)
    save_image(ORIGINALS / "case1_document.png", image)
    save_image(ORIGINALS / "case1_document_grayscale.png", grayscale)
    save_image(RESULTS / "case1_laplacian.png", laplacian)
    save_image(RESULTS / "case1_unsharp.png", unsharp)
    plot_grid("Case 1: Document sharpening",
              [("Original", image), ("Original grayscale", grayscale),
               ("Laplacian sharpened", laplacian),
               ("Unsharp mask", unsharp)],
              FIGURES / "case1_sharpening.png", columns=2, cmap="gray")


def run_case_2(input_dir: Path) -> None:
    image = gray(load_image("case2_medical", data.cell, input_dir))
    noisy = add_salt_pepper_noise(image, 0.06)
    mean = cv2.blur(noisy, (5, 5))
    gaussian = cv2.GaussianBlur(noisy, (5, 5), 1.2)
    median = cv2.medianBlur(noisy, 5)
    save_image(ORIGINALS / "case2_medical_clean.png", image)
    save_image(RESULTS / "case2_salt_pepper_noisy.png", noisy)
    for name, result in (("mean", mean), ("gaussian", gaussian), ("median", median)):
        save_image(RESULTS / f"case2_{name}.png", result)
    plot_grid("Case 2: Salt-and-pepper noise removal",
              [("Clean microscopy sample", image), ("Noisy input", noisy),
               ("Mean 5x5", mean), ("Gaussian 5x5", gaussian),
               ("Median 5x5", median)],
              FIGURES / "case2_denoising.png", columns=5, cmap="gray")


def run_case_3(input_dir: Path) -> None:
    image = load_image("case3_road", lambda: data.stereo_motorcycle()[0], input_dir)
    gray_image = gray(image)
    smoothed = cv2.GaussianBlur(gray_image, (5, 5), 1.0)
    sobel = sobel_edges(smoothed)
    prewitt = prewitt_edges(smoothed)
    save_image(ORIGINALS / "case3_road.png", image)
    save_image(RESULTS / "case3_sobel.png", sobel)
    save_image(RESULTS / "case3_prewitt.png", prewitt)
    plot_grid("Case 3: Sobel and Prewitt edge detection",
              [("Input image", image), ("Smoothed grayscale", smoothed),
               ("Sobel magnitude", sobel), ("Prewitt magnitude", prewitt)],
              FIGURES / "case3_road_edges.png", columns=2, cmap="gray")


def run_case_4(input_dir: Path) -> None:
    image = load_image("case4_cctv", data.rocket, input_dir)
    noisy = add_gaussian_noise(image, 20.0)
    gaussian = cv2.GaussianBlur(noisy, (5, 5), 1.2)
    bilateral = cv2.bilateralFilter(noisy, 9, 60, 60)
    save_image(ORIGINALS / "case4_outdoor.png", image)
    save_image(RESULTS / "case4_gaussian_noisy_input.png", noisy)
    save_image(RESULTS / "case4_gaussian_filtered.png", gaussian)
    save_image(RESULTS / "case4_bilateral_filtered.png", bilateral)
    plot_grid("Case 4: Filtering a noisy outdoor image",
              [("Original", image), ("Gaussian-noisy input", noisy),
               ("Gaussian filter 5x5", gaussian), ("Bilateral filter", bilateral)],
              FIGURES / "case4_outdoor_denoising.png", columns=2)


def run_case_5(input_dir: Path) -> None:
    # lily() is a public flower-and-foliage photograph; grass() is a fallback
    # for older scikit-image releases. Supply case5_leaf.* for a leaf close-up.
    sample_factory = getattr(data, "lily", data.grass)
    image = load_image("case5_leaf", sample_factory, input_dir)
    gray_image = gray(image)
    smoothed = cv2.GaussianBlur(gray_image, (5, 5), 1.0)
    sobel = sobel_edges(smoothed)
    laplacian = cv2.convertScaleAbs(cv2.Laplacian(smoothed, cv2.CV_32F, ksize=3))
    save_image(ORIGINALS / "case5_leaf_or_foliage.png", image)
    save_image(RESULTS / "case5_sobel.png", sobel)
    save_image(RESULTS / "case5_laplacian.png", laplacian)
    plot_grid("Case 5: Sobel and Laplacian leaf edges",
              [("Input photograph", image), ("Smoothed grayscale", smoothed),
               ("Sobel magnitude", sobel), ("Laplacian response", laplacian)],
              FIGURES / "case5_leaf_edges.png", columns=2, cmap="gray")


def run_case_6(input_dir: Path) -> None:
    image = load_image("case6_old_photo", data.camera, input_dir)
    if image.ndim == 3:
        image = gray(image)
    noisy = add_gaussian_noise(image, 12.0)
    median = cv2.medianBlur(noisy, 3)
    gaussian = cv2.GaussianBlur(noisy, (5, 5), 1.0)
    enhanced = unsharp_mask(median, amount=1.1)
    save_image(ORIGINALS / "case6_old_photo.png", image)
    save_image(RESULTS / "case6_noisy.png", noisy)
    save_image(RESULTS / "case6_median_denoised.png", median)
    save_image(RESULTS / "case6_gaussian_denoised.png", gaussian)
    save_image(RESULTS / "case6_preferred_enhanced.png", enhanced)
    plot_grid("Case 6: Denoising followed by controlled sharpening",
              [("Original", image), ("Synthetic grain", noisy),
               ("Median denoised", median), ("Gaussian denoised", gaussian),
               ("Median + unsharp mask", enhanced)],
              FIGURES / "case6_old_photo.png", columns=3, cmap="gray")


def run_case_7(input_dir: Path) -> None:
    image = gray(load_image("case7_product", data.coins, input_dir))
    panels = [("Original", image)]
    for size in (3, 5, 7):
        minimum = cv2.erode(image, np.ones((size, size), np.uint8))
        maximum = cv2.dilate(image, np.ones((size, size), np.uint8))
        save_image(RESULTS / f"case7_minimum_{size}x{size}.png", minimum)
        save_image(RESULTS / f"case7_maximum_{size}x{size}.png", maximum)
        panels.extend(((f"Minimum {size}x{size}", minimum),
                       (f"Maximum {size}x{size}", maximum)))
    save_image(ORIGINALS / "case7_product.png", image)
    plot_grid("Case 7: Minimum (erosion) and maximum (dilation) filters",
              panels, FIGURES / "case7_min_max_filters.png",
              columns=4, cmap="gray")


def run_case_8(input_dir: Path) -> None:
    clean = load_image("case8_clean", data.astronaut, input_dir)
    gaussian_noisy = add_gaussian_noise(clean, 18.0)
    impulse_noisy = add_salt_pepper_noise(clean, 0.04)
    rows: list[dict[str, str | float]] = []
    filter_results: dict[str, dict[str, np.ndarray]] = {}
    for noise_name, noisy in (("Gaussian", gaussian_noisy),
                              ("Salt-and-pepper", impulse_noisy)):
        filter_results[noise_name] = spatial_filters(noisy)
        for filter_name, result in filter_results[noise_name].items():
            mse, psnr = mse_psnr(clean, result)
            rows.append({"Noise": noise_name, "Filter": filter_name,
                         "MSE": mse, "PSNR_dB": psnr})
            save_image(RESULTS / f"case8_{noise_name.lower().replace('-', '_')}_"
                                 f"{filter_name.lower()}.png", result)

    save_image(ORIGINALS / "case8_clean_sample.png", clean)
    save_image(RESULTS / "case8_gaussian_noisy.png", gaussian_noisy)
    save_image(RESULTS / "case8_salt_pepper_noisy.png", impulse_noisy)
    csv_path = OUTPUT / "case8_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("Noise", "Filter", "MSE", "PSNR_dB"))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(FILTER_NAMES))
    for ax, noise_name in zip(axes, ("Gaussian", "Salt-and-pepper")):
        values = [next(row["PSNR_dB"] for row in rows
                       if row["Noise"] == noise_name and row["Filter"] == name)
                  for name in FILTER_NAMES]
        ax.bar(x, values, color=("#4472C4", "#70AD47", "#ED7D31", "#A5A5A5"))
        ax.set_xticks(x, FILTER_NAMES, rotation=20)
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(noise_name + " noise")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Case 8: Denoising comparison (higher PSNR is better)")
    fig.tight_layout()
    fig.savefig(FIGURES / "case8_psnr_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    panels = [("Clean", clean), ("Gaussian noise", gaussian_noisy)]
    panels.extend((f"Gaussian noise: {name}", result)
                  for name, result in filter_results["Gaussian"].items())
    panels.extend((f"Salt-pepper: {name}", result)
                  for name, result in filter_results["Salt-and-pepper"].items())
    plot_grid("Case 8: Visual comparison of spatial filters", panels,
              FIGURES / "case8_filter_results.png", columns=3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "inputs",
                        help="Directory for optional replacement images.")
    args = parser.parse_args()
    for directory in (ORIGINALS, RESULTS, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)

    for case in (run_case_1, run_case_2, run_case_3, run_case_4,
                 run_case_5, run_case_6, run_case_7, run_case_8):
        case(args.input_dir)
        print(f"Completed {case.__name__.replace('run_case_', 'Case ')}")
    print(f"Outputs saved under: {OUTPUT}")


if __name__ == "__main__":
    main()

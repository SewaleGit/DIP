from pathlib import Path

import cv2
import numpy as np
import streamlit as st

import spatial_filtering_project as project


ROOT = Path(__file__).resolve().parent


@st.cache_data
def read_upload(contents: bytes, name: str) -> np.ndarray:
    decoded = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError(f"Could not read uploaded image: {name}")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def sample_image(case: int) -> np.ndarray:
    sample_factory = getattr(project.data, "lily", project.data.grass)
    samples = {
        1: ("case1_document", project.data.page),
        2: ("case2_medical", project.data.cell),
        3: ("case3_road", lambda: project.data.stereo_motorcycle()[0]),
        4: ("case4_cctv", project.data.rocket),
        5: ("case5_leaf", sample_factory),
        6: ("case6_old_photo", project.data.camera),
        7: ("case7_product", project.data.coins),
        8: ("case8_clean", project.data.astronaut),
    }
    key, factory = samples[case]
    return project.load_image(key, factory, ROOT / "inputs")


def get_image(uploaded, case: int) -> np.ndarray:
    if uploaded is not None:
        return read_upload(uploaded.getvalue(), uploaded.name)
    return sample_image(case)


def show_images(panels: list[tuple[str, np.ndarray]], grayscale: bool = False,
                column_count: int | None = None) -> None:
    columns_per_row = column_count or min(3, len(panels))
    for start in range(0, len(panels), columns_per_row):
        columns = st.columns(columns_per_row)
        for column, (title, image) in zip(columns, panels[start:start + columns_per_row]):
            with column:
                st.image(image, caption=title, use_container_width=True,
                         clamp=True, channels="GRAY" if grayscale and image.ndim == 2 else "RGB")


st.set_page_config(
    page_title="Spatial Image Filtering",
    page_icon=":art:",
    layout="wide",
)

st.title("Spatial Image Filtering")
st.caption("Interactive demonstration of the eight image-processing cases")

with st.sidebar:
    st.header("Controls")
    case = st.selectbox(
        "Choose a case",
        range(1, 9),
        format_func=lambda value: f"Case {value}",
    )
    uploaded = st.file_uploader(
        "Upload an image (optional)",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
    )
    st.info("Without an upload, each case uses its matching public sample or file from inputs/.")

case_descriptions = {
    1: "Laplacian sharpening and unsharp masking",
    2: "Mean, Gaussian, and median salt-and-pepper denoising",
    3: "Sobel and Prewitt edge detection",
    4: "Gaussian and bilateral outdoor denoising",
    5: "Sobel and Laplacian leaf/foliage edge detection",
    6: "Old-photo denoising followed by controlled sharpening",
    7: "Minimum and maximum filtering at 3x3, 5x5, and 7x7",
    8: "Denoising comparison for Gaussian and salt-and-pepper noise",
}

try:
    image = get_image(uploaded, case)
except ValueError as error:
    st.error(str(error))
    st.stop()

st.subheader(f"Case {case}: {case_descriptions[case]}")

if case == 3:
    st.markdown(
        "**Case question:** Which operator produces clearer road boundaries "
        "in your selected image?"
    )
    has_road_input = any(
        (ROOT / "inputs" / f"case3_road{extension}").is_file()
        for extension in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    )
    if uploaded is None and has_road_input:
        st.success(
            "Using the included road photo with visible lane markings. "
            "In this image, Sobel traces the painted markings slightly more "
            "clearly; see the comparison below."
        )
    else:
        st.info(
            "For a meaningful road-boundary comparison, use an image with "
            "visible road markings. The bundled fallback, used when no "
            "case3_road image is present, is a motorcycle scene."
        )
elif case == 5:
    st.markdown(
        "**Case question:** Which operator best highlights the leaf outline, "
        "and what are its limitations?"
    )
    has_leaf_input = any(
        (ROOT / "inputs" / f"case5_leaf{extension}").is_file()
        for extension in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    )
    if uploaded is None and has_leaf_input:
        st.success(
            "Using the included isolated-leaf photo. Sobel shows the outline "
            "more clearly in this example; both operators also respond to "
            "internal veins."
        )
    else:
        st.info(
            "For a meaningful leaf-outline comparison, use a photo of an "
            "intact leaf with its outer edge visible against the background."
        )

if case == 1:
    gray = project.gray(image)
    panels = [
        ("Original", image),
        ("Original grayscale", gray),
        ("Laplacian sharpened", project.laplacian_sharpen(gray)),
        ("Unsharp mask", project.unsharp_mask(gray)),
    ]
    show_images(panels, grayscale=True, column_count=2)
    if st.button("Save four-panel PNG"):
        output_path = ROOT / "Case1" / "outputs" / "Compared.png"
        project.plot_grid(
            "Case 1: Document sharpening",
            panels,
            output_path,
            columns=4,
            cmap="gray",
        )
        st.success(f"Saved PNG to {output_path.relative_to(ROOT)}")

elif case == 2:
    gray = project.gray(image)
    noisy = project.add_salt_pepper_noise(gray, 0.06)
    show_images([
        ("Clean", gray),
        ("Salt-and-pepper noise", noisy),
        ("Mean 5x5", cv2.blur(noisy, (5, 5))),
        ("Gaussian 5x5", cv2.GaussianBlur(noisy, (5, 5), 1.2)),
        ("Median 5x5", cv2.medianBlur(noisy, 5)),
    ], grayscale=True, column_count=5)

elif case == 3:
    gray = project.gray(image)
    smoothed = cv2.GaussianBlur(gray, (5, 5), 1.0)
    show_images([
        ("Input image", image),
        ("Smoothed grayscale", smoothed),
        ("Sobel magnitude", project.sobel_edges(smoothed)),
        ("Prewitt magnitude", project.prewitt_edges(smoothed)),
    ], grayscale=True)

elif case == 4:
    noisy = project.add_gaussian_noise(image, 20.0)
    show_images([
        ("Original", image),
        ("Gaussian noise", noisy),
        ("Gaussian filtered", cv2.GaussianBlur(noisy, (5, 5), 1.2)),
        ("Bilateral filtered", cv2.bilateralFilter(noisy, 9, 60, 60)),
    ])

elif case == 5:
    gray = project.gray(image)
    smoothed = cv2.GaussianBlur(gray, (5, 5), 1.0)
    laplacian = cv2.convertScaleAbs(cv2.Laplacian(smoothed, cv2.CV_32F, ksize=3))
    show_images([
        ("Input photograph", image),
        ("Smoothed grayscale", smoothed),
        ("Sobel magnitude", project.sobel_edges(smoothed)),
        ("Laplacian response", laplacian),
    ], grayscale=True)

elif case == 6:
    gray = project.gray(image)
    noisy = project.add_gaussian_noise(gray, 12.0)
    median = cv2.medianBlur(noisy, 3)
    gaussian = cv2.GaussianBlur(noisy, (5, 5), 1.0)
    show_images([
        ("Original", gray),
        ("Synthetic grain", noisy),
        ("Median denoised", median),
        ("Gaussian denoised", gaussian),
        ("Median + unsharp", project.unsharp_mask(median, amount=1.1)),
    ], grayscale=True)

elif case == 7:
    gray = project.gray(image)
    panels = [("Original", gray)]
    for size in (3, 5, 7):
        kernel = np.ones((size, size), np.uint8)
        panels.append((f"Minimum {size}x{size}", cv2.erode(gray, kernel)))
        panels.append((f"Maximum {size}x{size}", cv2.dilate(gray, kernel)))
    show_images(panels, grayscale=True)

else:
    clean = image
    gaussian_noisy = project.add_gaussian_noise(clean, 18.0)
    impulse_noisy = project.add_salt_pepper_noise(clean, 0.04)

    rows = []
    panels = [
        ("Clean", clean),
        ("Gaussian noise", gaussian_noisy),
        ("Salt-and-pepper noise", impulse_noisy),
    ]
    for noise_name, noisy in (
        ("Gaussian", gaussian_noisy),
        ("Salt-and-pepper", impulse_noisy),
    ):
        for filter_name, result in project.spatial_filters(noisy).items():
            mse, psnr = project.mse_psnr(clean, result)
            rows.append({
                "Noise": noise_name,
                "Filter": filter_name,
                "MSE": round(mse, 4),
                "PSNR (dB)": round(psnr, 4),
            })
            panels.append((f"{noise_name}: {filter_name}", result))

    show_images(panels)
    st.subheader("MSE and PSNR metrics")
    st.dataframe(rows, use_container_width=True, hide_index=True)

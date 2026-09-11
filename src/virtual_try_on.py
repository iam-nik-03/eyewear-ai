from pathlib import Path
import math

import cv2
import numpy as np
from PIL import Image, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# MEDIAPIPE EYE LANDMARKS
# ============================================================

LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133

RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263


# ============================================================
# FRAME CALIBRATION
# ============================================================

# The visible glasses should normally be wider than the
# distance between the eyes.
#
# 2.20 means:
#
#       frame width ≈ eye distance × 2.20
#
# This can be tuned later for the individual catalogue.

FRAME_WIDTH_RATIO = 2.20

# Small vertical adjustment relative to eye distance.
FRAME_VERTICAL_OFFSET = 0.03

# Pixels used to feather the generated alpha mask.
ALPHA_BLUR_RADIUS = 0.7


# ============================================================
# LANDMARK → PIXEL
# ============================================================

def landmark_to_pixel(
    landmark,
    image_width,
    image_height,
):
    """
    Convert normalized MediaPipe coordinates
    into image pixel coordinates.
    """

    return np.array(
        [
            landmark.x * image_width,
            landmark.y * image_height,
        ],
        dtype=np.float32,
    )


# ============================================================
# EYE GEOMETRY
# ============================================================

def get_eye_geometry(
    landmarks,
    image_width,
    image_height,
):
    """
    Calculate:

    - left eye center
    - right eye center
    - eye-line center
    - inter-eye distance
    - face rotation angle
    """

    left_outer = landmark_to_pixel(
        landmarks[LEFT_EYE_OUTER],
        image_width,
        image_height,
    )

    left_inner = landmark_to_pixel(
        landmarks[LEFT_EYE_INNER],
        image_width,
        image_height,
    )

    right_inner = landmark_to_pixel(
        landmarks[RIGHT_EYE_INNER],
        image_width,
        image_height,
    )

    right_outer = landmark_to_pixel(
        landmarks[RIGHT_EYE_OUTER],
        image_width,
        image_height,
    )

    left_eye_center = (
        left_outer + left_inner
    ) / 2.0

    right_eye_center = (
        right_outer + right_inner
    ) / 2.0

    eye_vector = (
        right_eye_center
        - left_eye_center
    )

    eye_distance = float(
        np.linalg.norm(eye_vector)
    )

    angle = math.degrees(
        math.atan2(
            eye_vector[1],
            eye_vector[0],
        )
    )

    eye_center = (
        left_eye_center
        + right_eye_center
    ) / 2.0

    return {
        "left_eye_center": left_eye_center,
        "right_eye_center": right_eye_center,
        "eye_center": eye_center,
        "eye_distance": eye_distance,
        "angle": angle,
    }


# ============================================================
# BACKGROUND ESTIMATION
# ============================================================

def estimate_background_color(image):
    """
    Estimate background colour from the four corners.

    Product images normally place the frame against a
    relatively uniform background.
    """

    array = np.asarray(
        image.convert("RGB")
    )

    height, width = array.shape[:2]

    sample_size = max(
        10,
        int(min(height, width) * 0.08),
    )

    samples = np.concatenate(
        [
            array[
                :sample_size,
                :sample_size,
            ].reshape(-1, 3),

            array[
                :sample_size,
                width - sample_size:
            ].reshape(-1, 3),

            array[
                height - sample_size:,
                :sample_size,
            ].reshape(-1, 3),

            array[
                height - sample_size:,
                width - sample_size:
            ].reshape(-1, 3),
        ],
        axis=0,
    )

    return np.median(
        samples,
        axis=0,
    )


# ============================================================
# FRAME MASK
# ============================================================

def create_frame_mask(
    image,
    background_threshold=42,
):
    """
    Generate an approximate foreground mask.

    This is deliberately designed for catalogue-style
    product photographs where the glasses are centred
    against a relatively simple background.
    """

    rgb = np.asarray(
        image.convert("RGB")
    )

    background = estimate_background_color(
        image
    )

    difference = np.linalg.norm(
        rgb.astype(np.float32)
        - background.astype(np.float32),
        axis=2,
    )

    mask = np.where(
        difference > background_threshold,
        255,
        0,
    ).astype(np.uint8)

    # Remove isolated noise.
    kernel = np.ones(
        (5, 5),
        dtype=np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    # Keep the connected components that contain
    # the central catalogue object.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    if num_labels <= 1:
        return mask

    height, width = mask.shape

    centre_x = width / 2.0
    centre_y = height / 2.0

    candidates = []

    for label in range(1, num_labels):

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]

        component_width = stats[
            label,
            cv2.CC_STAT_WIDTH,
        ]

        component_height = stats[
            label,
            cv2.CC_STAT_HEIGHT,
        ]

        area = stats[
            label,
            cv2.CC_STAT_AREA,
        ]

        component_centre_x = (
            x + component_width / 2
        )

        component_centre_y = (
            y + component_height / 2
        )

        centre_distance = math.sqrt(
            (
                component_centre_x
                - centre_x
            ) ** 2
            +
            (
                component_centre_y
                - centre_y
            ) ** 2
        )

        # Prefer large components near the image centre.
        score = (
            area
            / max(centre_distance, 1.0)
        )

        candidates.append(
            (score, label)
        )

    candidates.sort(
        reverse=True
    )

    selected_labels = [
        label
        for _, label in candidates[:3]
    ]

    clean_mask = np.zeros_like(
        mask
    )

    for label in selected_labels:

        clean_mask[
            labels == label
        ] = 255

    return clean_mask


# ============================================================
# LENS INTERIOR REMOVAL
# ============================================================

def remove_lens_interior(
    image,
    mask,
):
    """
    Remove large interior regions inside the glasses.

    This prevents a white catalogue background from appearing
    as opaque rectangles where the transparent lenses should be.
    """

    result_mask = mask.copy()

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return result_mask

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True,
    )

    for contour in contours[:4]:

        area = cv2.contourArea(
            contour
        )

        if area <= 0:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        # Search for enclosed bright/low-detail regions.
        roi = result_mask[
            y:y + h,
            x:x + w,
        ]

        if roi.size == 0:
            continue

        # Fill holes inside the outer contour.
        filled = np.zeros_like(
            roi
        )

        cv2.drawContours(
            filled,
            [
                contour
                - np.array(
                    [[[x, y]]]
                )
            ],
            -1,
            255,
            thickness=cv2.FILLED,
        )

        holes = cv2.bitwise_xor(
            filled,
            roi,
        )

        # Only remove relatively large interior holes.
        hole_pixels = np.count_nonzero(
            holes
        )

        if (
            hole_pixels
            > 0.08 * roi.size
        ):
            roi[
                holes > 0
            ] = 0

    return result_mask


# ============================================================
# ALPHA MASK
# ============================================================

def build_alpha_mask(
    image,
):
    """
    Build a soft RGBA alpha mask from the catalogue image.
    """

    mask = create_frame_mask(
        image
    )

    mask = remove_lens_interior(
        image,
        mask,
    )

    # Slight edge smoothing.
    alpha_image = Image.fromarray(
        mask
    )

    if ALPHA_BLUR_RADIUS > 0:
        alpha_image = alpha_image.filter(
            ImageFilter.GaussianBlur(
                ALPHA_BLUR_RADIUS
            )
        )

    return np.array(
        alpha_image,
        dtype=np.uint8,
        copy=True,
    )


# ============================================================
# CROP FRAME
# ============================================================

def crop_to_frame(
    image,
    alpha,
    padding_ratio=0.08,
):
    """
    Crop away the large catalogue background.

    This is the major improvement over V1:
    we scale the ACTUAL glasses instead of scaling
    the complete 1080×1080 product photograph.
    """

    non_zero = np.where(
        alpha > 12
    )

    if len(non_zero[0]) == 0:
        raise ValueError(
            "Could not isolate the eyewear from the catalogue image."
        )

    y_min = int(
        non_zero[0].min()
    )

    y_max = int(
        non_zero[0].max()
    )

    x_min = int(
        non_zero[1].min()
    )

    x_max = int(
        non_zero[1].max()
    )

    height = y_max - y_min + 1
    width = x_max - x_min + 1

    padding_x = int(
        width * padding_ratio
    )

    padding_y = int(
        height * padding_ratio
    )

    x_min = max(
        0,
        x_min - padding_x,
    )

    y_min = max(
        0,
        y_min - padding_y,
    )

    x_max = min(
        image.width - 1,
        x_max + padding_x,
    )

    y_max = min(
        image.height - 1,
        y_max + padding_y,
    )

    image_crop = image.crop(
        (
            x_min,
            y_min,
            x_max + 1,
            y_max + 1,
        )
    )

    alpha_crop = alpha[
        y_min:y_max + 1,
        x_min:x_max + 1,
    ]

    return (
        image_crop,
        alpha_crop,
    )


# ============================================================
# PREPARE FRAME
# ============================================================

def prepare_frame(
    frame_path,
):
    """
    Load a catalogue JPEG and convert it into
    a cropped RGBA eyewear asset.
    """

    frame_path = Path(
        frame_path
    )

    if not frame_path.exists():
        raise FileNotFoundError(
            f"Frame image not found:\n{frame_path}"
        )

    image = Image.open(
        frame_path
    ).convert("RGB")

    alpha = build_alpha_mask(
        image
    )

    cropped_image, cropped_alpha = (
        crop_to_frame(
            image,
            alpha,
        )
    )

    # Make an explicit writable NumPy array before assigning alpha.
    rgba = np.array(
        cropped_image.convert("RGBA"),
        copy=True,
    )

    rgba[:, :, 3] = cropped_alpha

    return Image.fromarray(
        rgba
    )


# ============================================================
# RESIZE FRAME
# ============================================================

def resize_frame(
    frame,
    target_width,
):
    """
    Resize based on the actual visible frame width.
    """

    width, height = frame.size

    if width <= 0:
        raise ValueError(
            "Invalid frame width."
        )

    scale = (
        target_width / width
    )

    target_height = max(
        1,
        int(height * scale),
    )

    return frame.resize(
        (
            int(target_width),
            target_height,
        ),
        Image.Resampling.LANCZOS,
    )


# ============================================================
# ROTATE FRAME
# ============================================================

def rotate_frame(
    frame,
    angle,
):
    """
    Rotate around the frame centre.
    """

    return frame.rotate(
        angle,
        expand=True,
        resample=Image.Resampling.BICUBIC,
    )


# ============================================================
# ALPHA COMPOSITING
# ============================================================

def overlay_transparent(
    background,
    overlay,
    center_x,
    center_y,
):
    """
    Alpha-composite an RGBA frame onto the user's image.
    """

    background = background.convert(
        "RGBA"
    )

    overlay = overlay.convert(
        "RGBA"
    )

    x = int(
        center_x
        - overlay.width / 2
    )

    y = int(
        center_y
        - overlay.height / 2
    )

    background.alpha_composite(
        overlay,
        (
            x,
            y,
        ),
    )

    return background


# ============================================================
# MAIN VIRTUAL TRY-ON
# ============================================================

def apply_virtual_try_on(
    image,
    landmarks,
    frame_path,
    frame_width_ratio=FRAME_WIDTH_RATIO,
):
    """
    Apply a catalogue frame to a user's face.

    The algorithm uses:

        1. MediaPipe eye landmarks
        2. Inter-eye distance
        3. Eye-line rotation
        4. Cropped frame dimensions
        5. Alpha transparency
        6. Geometric scaling
        7. Rotation
        8. Position alignment

    This is a deterministic CV-based try-on,
    not generative image synthesis.
    """

    image = image.convert(
        "RGB"
    )

    image_width, image_height = (
        image.size
    )

    if len(landmarks) < 478:
        raise ValueError(
            "Expected 478 MediaPipe landmarks."
        )

    geometry = get_eye_geometry(
        landmarks,
        image_width,
        image_height,
    )

    eye_distance = geometry[
        "eye_distance"
    ]

    if eye_distance <= 1:
        raise ValueError(
            "Invalid eye distance detected."
        )

    eye_center = geometry[
        "eye_center"
    ]

    angle = geometry[
        "angle"
    ]

    # --------------------------------------------------------
    # Prepare actual visible frame
    # --------------------------------------------------------

    frame = prepare_frame(
        frame_path
    )

    # --------------------------------------------------------
    # Calculate target frame width
    # --------------------------------------------------------

    target_width = (
        eye_distance
        * frame_width_ratio
    )

    frame = resize_frame(
        frame,
        target_width,
    )

    # --------------------------------------------------------
    # Rotate with face
    # --------------------------------------------------------

    frame = rotate_frame(
        frame,
        -angle,
    )

    # --------------------------------------------------------
    # Position
    # --------------------------------------------------------

    center_x = float(
        eye_center[0]
    )

    center_y = float(
        eye_center[1]
        + eye_distance
        * FRAME_VERTICAL_OFFSET
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    result = overlay_transparent(
        image,
        frame,
        center_x,
        center_y,
    )

    return (
        result.convert("RGB"),
        geometry,
    )


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def try_on_frame(
    image_path,
    frame_path,
    landmarks,
    output_path=None,
):
    """
    Convenience wrapper for CLI/testing.
    """

    image = Image.open(
        image_path
    ).convert("RGB")

    result, geometry = (
        apply_virtual_try_on(
            image=image,
            landmarks=landmarks,
            frame_path=frame_path,
        )
    )

    if output_path is not None:

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        result.save(
            output_path,
            quality=95,
        )

    return (
        result,
        geometry,
    )
from pathlib import Path
import csv

import cv2
import mediapipe as mp


# =========================================================
# CONFIGURATION
# =========================================================

DATASET_PATH = Path(
    r"C:\Users\x03xN\.cache\kagglehub\datasets\niten19\face-shape-dataset\versions\2\FaceShape Dataset"
)

MODEL_PATH = Path(
    "models/mediapipe/face_landmarker.task"
)

OUTPUT_DIR = Path(
    "data/processed/faces"
)

FEATURES_PATH = (
    OUTPUT_DIR / "facial_features.csv"
)

FAILURES_PATH = (
    OUTPUT_DIR / "processing_failures.csv"
)

CLASSES = [
    "Heart",
    "Oblong",
    "Oval",
    "Round",
    "Square",
]

SPLITS = [
    "training_set",
    "testing_set",
]


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def distance(point_a, point_b):
    """Calculate normalized Euclidean distance."""

    return (
        (point_a.x - point_b.x) ** 2
        +
        (point_a.y - point_b.y) ** 2
    ) ** 0.5


def get_point(landmarks, index):
    """Return a landmark safely."""

    if index >= len(landmarks):
        raise ValueError(
            f"Landmark index {index} unavailable."
        )

    return landmarks[index]


def extract_features(face):
    """Extract facial geometry features."""

    # -----------------------------------------------------
    # Landmark references
    # -----------------------------------------------------

    forehead_top = get_point(face, 10)

    chin = get_point(face, 152)

    left_face = get_point(face, 234)

    right_face = get_point(face, 454)

    left_cheek = get_point(face, 93)

    right_cheek = get_point(face, 323)

    left_jaw = get_point(face, 172)

    right_jaw = get_point(face, 397)

    left_forehead = get_point(face, 54)

    right_forehead = get_point(face, 284)

    # -----------------------------------------------------
    # Measurements
    # -----------------------------------------------------

    face_height = distance(
        forehead_top,
        chin
    )

    face_width = distance(
        left_face,
        right_face
    )

    cheek_width = distance(
        left_cheek,
        right_cheek
    )

    jaw_width = distance(
        left_jaw,
        right_jaw
    )

    forehead_width = distance(
        left_forehead,
        right_forehead
    )

    # -----------------------------------------------------
    # Safety check
    # -----------------------------------------------------

    if face_height <= 0:
        raise ValueError(
            "Invalid face height."
        )

    if face_width <= 0:
        raise ValueError(
            "Invalid face width."
        )

    # -----------------------------------------------------
    # Ratios
    # -----------------------------------------------------

    face_aspect_ratio = (
        face_width / face_height
    )

    forehead_to_face_ratio = (
        forehead_width / face_width
    )

    cheek_to_face_ratio = (
        cheek_width / face_width
    )

    jaw_to_face_ratio = (
        jaw_width / face_width
    )

    # -----------------------------------------------------
    # Feature vector
    # -----------------------------------------------------

    return {
        "face_width": face_width,
        "face_height": face_height,
        "face_aspect_ratio": face_aspect_ratio,
        "forehead_width": forehead_width,
        "cheek_width": cheek_width,
        "jaw_width": jaw_width,
        "forehead_to_face_ratio":
            forehead_to_face_ratio,
        "cheek_to_face_ratio":
            cheek_to_face_ratio,
        "jaw_to_face_ratio":
            jaw_to_face_ratio,
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("EYewear AI - BUILD FACIAL FEATURE DATASET")
    print("=" * 70)

    # -----------------------------------------------------
    # Validate model
    # -----------------------------------------------------

    if not MODEL_PATH.exists():

        print(
            "\nERROR: Face Landmarker model not found:"
        )

        print(
            MODEL_PATH
        )

        return

    # -----------------------------------------------------
    # Prepare output
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    feature_rows = []
    failure_rows = []

    total_images = 0
    successful_images = 0
    failed_images = 0

    # -----------------------------------------------------
    # Create MediaPipe detector ONCE
    # -----------------------------------------------------

    BaseOptions = mp.tasks.BaseOptions

    FaceLandmarker = (
        mp.tasks.vision.FaceLandmarker
    )

    FaceLandmarkerOptions = (
        mp.tasks.vision.FaceLandmarkerOptions
    )

    RunningMode = (
        mp.tasks.vision.RunningMode
    )

    base_options = BaseOptions(
        model_asset_path=str(
            MODEL_PATH
        ),
        delegate=BaseOptions.Delegate.CPU
    )

    options = FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,
        num_faces=1
    )

    print("\nStarting feature extraction...\n")

    # -----------------------------------------------------
    # Process all images
    # -----------------------------------------------------

    with FaceLandmarker.create_from_options(
        options
    ) as landmarker:

        for split in SPLITS:

            for class_name in CLASSES:

                class_path = (
                    DATASET_PATH
                    / split
                    / class_name
                )

                if not class_path.exists():

                    print(
                        f"WARNING: Missing folder: "
                        f"{class_path}"
                    )

                    continue

                image_files = [
                    file
                    for file in class_path.iterdir()
                    if file.is_file()
                    and file.suffix.lower()
                    in {
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".webp",
                        ".bmp",
                    }
                ]

                print(
                    f"{split} / {class_name}: "
                    f"{len(image_files)} images"
                )

                for image_path in image_files:

                    total_images += 1

                    try:

                        # -----------------------------
                        # Read image
                        # -----------------------------

                        image = cv2.imread(
                            str(image_path)
                        )

                        if image is None:

                            raise ValueError(
                                "OpenCV could not read image."
                            )

                        # -----------------------------
                        # BGR → RGB
                        # -----------------------------

                        rgb_image = cv2.cvtColor(
                            image,
                            cv2.COLOR_BGR2RGB
                        )

                        # -----------------------------
                        # MediaPipe image
                        # -----------------------------

                        mp_image = mp.Image(
                            image_format=
                                mp.ImageFormat.SRGB,
                            data=rgb_image
                        )

                        # -----------------------------
                        # Detect landmarks
                        # -----------------------------

                        result = landmarker.detect(
                            mp_image
                        )

                        if not result.face_landmarks:

                            raise ValueError(
                                "No face detected."
                            )

                        # -----------------------------
                        # First face
                        # -----------------------------

                        face = (
                            result.face_landmarks[0]
                        )

                        # -----------------------------
                        # Extract features
                        # -----------------------------

                        features = (
                            extract_features(face)
                        )

                        # -----------------------------
                        # Store result
                        # -----------------------------

                        row = {
                            "image":
                                image_path.name,

                            "split":
                                split,

                            "label":
                                class_name,

                            **features,
                        }

                        feature_rows.append(row)

                        successful_images += 1

                    except Exception as error:

                        failure_rows.append({

                            "image":
                                image_path.name,

                            "split":
                                split,

                            "label":
                                class_name,

                            "reason":
                                str(error),
                        })

                        failed_images += 1

                    # Progress
                    if total_images % 100 == 0:

                        print(
                            f"Processed: "
                            f"{total_images:,} | "
                            f"Success: "
                            f"{successful_images:,} | "
                            f"Failed: "
                            f"{failed_images:,}"
                        )

    # =====================================================
    # SAVE FEATURES
    # =====================================================

    if feature_rows:

        feature_fields = [
            "image",
            "split",
            "label",
            "face_width",
            "face_height",
            "face_aspect_ratio",
            "forehead_width",
            "cheek_width",
            "jaw_width",
            "forehead_to_face_ratio",
            "cheek_to_face_ratio",
            "jaw_to_face_ratio",
        ]

        with open(
            FEATURES_PATH,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=feature_fields
            )

            writer.writeheader()

            writer.writerows(
                feature_rows
            )

    # =====================================================
    # SAVE FAILURES
    # =====================================================

    failure_fields = [
        "image",
        "split",
        "label",
        "reason",
    ]

    with open(
        FAILURES_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=failure_fields
        )

        writer.writeheader()

        writer.writerows(
            failure_rows
        )

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION FINISHED")
    print("=" * 70)

    print(
        f"\nTotal images : {total_images:,}"
    )

    print(
        f"Successful   : {successful_images:,}"
    )

    print(
        f"Failed       : {failed_images:,}"
    )

    if total_images > 0:

        success_rate = (
            successful_images
            / total_images
            * 100
        )

        print(
            f"Success rate : {success_rate:.2f}%"
        )

    print(
        "\nFeatures saved to:"
    )

    print(
        FEATURES_PATH.resolve()
    )

    print(
        "\nFailures saved to:"
    )

    print(
        FAILURES_PATH.resolve()
    )


if __name__ == "__main__":
    main()
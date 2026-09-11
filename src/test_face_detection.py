from pathlib import Path

import cv2
import mediapipe as mp


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

DATASET_PATH = Path(
    r"C:\Users\x03xN\.cache\kagglehub\datasets\niten19\face-shape-dataset\versions\2\FaceShape Dataset"
)

MODEL_PATH = Path(
    "models/mediapipe/blaze_face_short_range.tflite"
)

OUTPUT_PATH = Path(
    "data/face_detection_test.jpg"
)


# ---------------------------------------------------------
# FIND ONE IMAGE
# ---------------------------------------------------------

heart_images = list(
    (DATASET_PATH / "training_set" / "Heart").rglob("*.jpg")
)

if not heart_images:

    print("ERROR: No Heart JPG images found.")
    raise SystemExit

image_path = heart_images[0]


# ---------------------------------------------------------
# CHECK PATHS
# ---------------------------------------------------------

if not image_path.exists():

    print("ERROR: Image not found:")
    print(image_path)

    raise SystemExit


if not MODEL_PATH.exists():

    print("ERROR: MediaPipe model not found:")
    print(MODEL_PATH)

    raise SystemExit


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

print("=" * 70)
print("EYewear AI - FACE DETECTION TEST")
print("=" * 70)

print("\nInput image:")
print(image_path)

print("\nMediaPipe model:")
print(MODEL_PATH)


# ---------------------------------------------------------
# READ IMAGE
# ---------------------------------------------------------

image = cv2.imread(
    str(image_path)
)

if image is None:

    print("\nERROR: OpenCV could not read the image.")

    raise SystemExit


height, width = image.shape[:2]

print("\nImage loaded successfully.")
print(f"Image size: {width} x {height}")


# ---------------------------------------------------------
# CONVERT BGR → RGB
# ---------------------------------------------------------

rgb_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# ---------------------------------------------------------
# CREATE MEDIAPIPE IMAGE
# ---------------------------------------------------------

mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb_image
)


# ---------------------------------------------------------
# CREATE FACE DETECTOR
# ---------------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions

base_options = BaseOptions(
    model_asset_path=str(MODEL_PATH),
    delegate=BaseOptions.Delegate.CPU
)

options = FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.5
)


# ---------------------------------------------------------
# RUN FACE DETECTION
# ---------------------------------------------------------

with FaceDetector.create_from_options(options) as detector:

    result = detector.detect(mp_image)


# ---------------------------------------------------------
# CHECK RESULTS
# ---------------------------------------------------------

detections = result.detections

if not detections:

    print("\n❌ No face detected.")

else:

    print(
        f"\n✅ Faces detected: {len(detections)}"
    )

    for index, detection in enumerate(
        detections,
        start=1
    ):

        bounding_box = detection.bounding_box

        x = bounding_box.origin_x
        y = bounding_box.origin_y

        box_width = bounding_box.width
        box_height = bounding_box.height

        # Keep coordinates inside image
        x = max(0, x)
        y = max(0, y)

        box_width = min(
            box_width,
            width - x
        )

        box_height = min(
            box_height,
            height - y
        )

        print(f"\nFace {index}:")
        print(f"  x         : {x}")
        print(f"  y         : {y}")
        print(f"  width     : {box_width}")
        print(f"  height    : {box_height}")

        # Detection confidence
        confidence = detection.categories[0].score

        print(
            f"  confidence: {confidence:.4f}"
        )

        # Draw bounding box
        cv2.rectangle(
            image,
            (x, y),
            (
                x + box_width,
                y + box_height
            ),
            (0, 255, 0),
            3
        )

        # Add confidence text
        cv2.putText(
            image,
            f"Face: {confidence:.2f}",
            (
                x,
                max(30, y - 10)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


# ---------------------------------------------------------
# SAVE RESULT
# ---------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

success = cv2.imwrite(
    str(OUTPUT_PATH),
    image
)


# ---------------------------------------------------------
# FINAL RESULT
# ---------------------------------------------------------

if success:

    print("\n" + "=" * 70)
    print("FACE DETECTION TEST COMPLETE")
    print("=" * 70)

    print("\nResult saved to:")

    print(
        OUTPUT_PATH.resolve()
    )

else:

    print(
        "\nERROR: Could not save output image."
    )
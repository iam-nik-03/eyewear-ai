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
    "models/mediapipe/face_landmarker.task"
)

OUTPUT_PATH = Path(
    "data/face_landmarks_test.jpg"
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

    print("ERROR: Face Landmarker model not found:")
    print(MODEL_PATH)

    raise SystemExit


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

print("=" * 70)
print("EYewear AI - FACE LANDMARK TEST")
print("=" * 70)

print("\nInput image:")
print(image_path)

print("\nModel:")
print(MODEL_PATH)


# ---------------------------------------------------------
# READ IMAGE
# ---------------------------------------------------------

image = cv2.imread(
    str(image_path)
)

if image is None:

    print("\nERROR: OpenCV could not read image.")

    raise SystemExit


height, width = image.shape[:2]

print("\nImage loaded successfully.")

print(
    f"Image size: {width} x {height}"
)


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
# CREATE FACE LANDMARKER
# ---------------------------------------------------------

BaseOptions = mp.tasks.BaseOptions

FaceLandmarker = (
    mp.tasks.vision.FaceLandmarker
)

FaceLandmarkerOptions = (
    mp.tasks.vision.FaceLandmarkerOptions
)

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)


base_options = BaseOptions(
    model_asset_path=str(MODEL_PATH),
    delegate=BaseOptions.Delegate.CPU
)


options = FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)


# ---------------------------------------------------------
# RUN LANDMARK DETECTION
# ---------------------------------------------------------

with FaceLandmarker.create_from_options(
    options
) as landmarker:

    result = landmarker.detect(
        mp_image
    )


# ---------------------------------------------------------
# CHECK RESULTS
# ---------------------------------------------------------

if not result.face_landmarks:

    print("\n❌ No face landmarks detected.")

    raise SystemExit


print(
    f"\n✅ Faces detected: "
    f"{len(result.face_landmarks)}"
)


# ---------------------------------------------------------
# DRAW LANDMARKS
# ---------------------------------------------------------

for face_index, face_landmarks in enumerate(
    result.face_landmarks,
    start=1
):

    print(
        f"\nFace {face_index}:"
    )

    print(
        f"  Number of landmarks: "
        f"{len(face_landmarks)}"
    )

    for landmark in face_landmarks:

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        # Keep point inside image
        x = max(
            0,
            min(x, width - 1)
        )

        y = max(
            0,
            min(y, height - 1)
        )

        cv2.circle(
            image,
            (x, y),
            1,
            (0, 255, 0),
            -1
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
    print("FACE LANDMARK TEST COMPLETE")
    print("=" * 70)

    print("\nResult saved to:")

    print(
        OUTPUT_PATH.resolve()
    )

else:

    print(
        "\nERROR: Could not save output."
    )
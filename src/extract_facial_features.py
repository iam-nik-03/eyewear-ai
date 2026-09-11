from pathlib import Path
import math
import csv

import cv2
import mediapipe as mp


# =========================================================
# PATHS
# =========================================================

DATASET_PATH = Path(
    r"C:\Users\x03xN\.cache\kagglehub\datasets\niten19\face-shape-dataset\versions\2\FaceShape Dataset"
)

MODEL_PATH = Path(
    "models/mediapipe/face_landmarker.task"
)

OUTPUT_PATH = Path(
    "data/facial_features_test.csv"
)


# =========================================================
# TEST IMAGE
# =========================================================

image_path = (
    DATASET_PATH
    / "training_set"
    / "Heart"
    / "heart (1).jpg"
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def distance(point_a, point_b):
    """
    Calculate Euclidean distance between
    two normalized facial landmark points.
    """

    return math.sqrt(
        (point_a.x - point_b.x) ** 2
        +
        (point_a.y - point_b.y) ** 2
    )


def get_point(landmarks, index):
    """
    Safely get one facial landmark.
    """

    if index >= len(landmarks):
        raise ValueError(
            f"Landmark index {index} does not exist."
        )

    return landmarks[index]


# =========================================================
# CHECK FILES
# =========================================================

if not image_path.exists():

    print("ERROR: Test image not found:")
    print(image_path)

    raise SystemExit


if not MODEL_PATH.exists():

    print("ERROR: Face Landmarker model not found:")
    print(MODEL_PATH)

    raise SystemExit


# =========================================================
# HEADER
# =========================================================

print("=" * 70)
print("EYewear AI - FACIAL GEOMETRY FEATURE EXTRACTION")
print("=" * 70)

print("\nInput image:")
print(image_path)


# =========================================================
# LOAD IMAGE
# =========================================================

image = cv2.imread(
    str(image_path)
)

if image is None:

    print("\nERROR: Could not read image.")

    raise SystemExit


height, width = image.shape[:2]

print(
    f"\nImage size: {width} x {height}"
)


# =========================================================
# CONVERT IMAGE
# =========================================================

rgb_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb_image
)


# =========================================================
# CREATE FACE LANDMARKER
# =========================================================

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
    model_asset_path=str(MODEL_PATH),
    delegate=BaseOptions.Delegate.CPU
)


options = FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.IMAGE,
    num_faces=1
)


# =========================================================
# RUN LANDMARK MODEL
# =========================================================

with FaceLandmarker.create_from_options(
    options
) as landmarker:

    result = landmarker.detect(
        mp_image
    )


# =========================================================
# CHECK FACE
# =========================================================

if not result.face_landmarks:

    print("\n❌ No face detected.")

    raise SystemExit


face = result.face_landmarks[0]

print(
    f"\n✅ Face detected"
)

print(
    f"Landmarks detected: {len(face)}"
)


# =========================================================
# LANDMARK POINTS
# =========================================================
#
# These landmark indices represent approximate
# facial regions in MediaPipe's face topology.
#
# We use them as geometric reference points rather
# than treating any single measurement as a face-shape rule.
#
# =========================================================

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


# =========================================================
# CALCULATE RAW GEOMETRIC MEASUREMENTS
# =========================================================

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


# =========================================================
# CALCULATE NORMALIZED RATIOS
# =========================================================

if face_height == 0:

    print("\nERROR: Face height is zero.")

    raise SystemExit


face_aspect_ratio = (
    face_width / face_height
)

cheek_to_face_ratio = (
    cheek_width / face_width
)

jaw_to_face_ratio = (
    jaw_width / face_width
)

forehead_to_face_ratio = (
    forehead_width / face_width
)


# =========================================================
# CREATE FEATURE RECORD
# =========================================================

features = {

    "image": image_path.name,

    "face_shape_label":
        image_path.parent.name,

    "face_width":
        face_width,

    "face_height":
        face_height,

    "face_aspect_ratio":
        face_aspect_ratio,

    "forehead_width":
        forehead_width,

    "cheek_width":
        cheek_width,

    "jaw_width":
        jaw_width,

    "forehead_to_face_ratio":
        forehead_to_face_ratio,

    "cheek_to_face_ratio":
        cheek_to_face_ratio,

    "jaw_to_face_ratio":
        jaw_to_face_ratio,
}


# =========================================================
# PRINT FEATURES
# =========================================================

print("\n" + "-" * 70)
print("FACIAL GEOMETRY FEATURES")
print("-" * 70)

for feature_name, value in features.items():

    if isinstance(value, float):

        print(
            f"{feature_name:30s}: "
            f"{value:.6f}"
        )

    else:

        print(
            f"{feature_name:30s}: "
            f"{value}"
        )


# =========================================================
# SAVE CSV
# =========================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=features.keys()
    )

    writer.writeheader()

    writer.writerow(
        features
    )


# =========================================================
# COMPLETE
# =========================================================

print("\n" + "=" * 70)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 70)

print("\nCSV saved to:")

print(
    OUTPUT_PATH.resolve()
)
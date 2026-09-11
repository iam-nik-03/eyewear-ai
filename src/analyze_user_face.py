from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "face_landmarks_test.jpg"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "face_shape"
    / "geometry_svm.joblib"
)

LANDMARK_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mediapipe"
    / "face_landmarker.task"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "faces"
    / "user_face_analysis.csv"
)


# ============================================================
# FEATURE SCHEMA
# ============================================================

FEATURE_COLUMNS = [
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


# ============================================================
# MEDIAPIPE LANDMARK INDICES
# ============================================================

FOREHEAD_TOP = 10
CHIN = 152

LEFT_FACE = 234
RIGHT_FACE = 454

LEFT_CHEEK = 93
RIGHT_CHEEK = 323

LEFT_JAW = 172
RIGHT_JAW = 397

LEFT_FOREHEAD = 54
RIGHT_FOREHEAD = 284


# ============================================================
# VIRTUAL TRY-ON EYE LANDMARKS
# ============================================================

LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133

RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263


# ============================================================
# GEOMETRY
# ============================================================

def distance(point_a, point_b):
    """
    Calculate Euclidean distance between two
    normalized MediaPipe landmarks.
    """

    dx = point_a.x - point_b.x
    dy = point_a.y - point_b.y

    return (dx * dx + dy * dy) ** 0.5


def extract_features(landmarks):
    """
    Extract the exact facial geometry features used
    during face-shape model training.
    """

    face_width = distance(
        landmarks[LEFT_FACE],
        landmarks[RIGHT_FACE],
    )

    face_height = distance(
        landmarks[FOREHEAD_TOP],
        landmarks[CHIN],
    )

    forehead_width = distance(
        landmarks[LEFT_FOREHEAD],
        landmarks[RIGHT_FOREHEAD],
    )

    cheek_width = distance(
        landmarks[LEFT_CHEEK],
        landmarks[RIGHT_CHEEK],
    )

    jaw_width = distance(
        landmarks[LEFT_JAW],
        landmarks[RIGHT_JAW],
    )

    if face_width == 0 or face_height == 0:
        raise ValueError(
            "Invalid facial geometry: zero face dimension."
        )

    return {
        "face_width": face_width,
        "face_height": face_height,
        "face_aspect_ratio": (
            face_width / face_height
        ),
        "forehead_width": forehead_width,
        "cheek_width": cheek_width,
        "jaw_width": jaw_width,
        "forehead_to_face_ratio": (
            forehead_width / face_width
        ),
        "cheek_to_face_ratio": (
            cheek_width / face_width
        ),
        "jaw_to_face_ratio": (
            jaw_width / face_width
        ),
    }


# ============================================================
# MODEL VALIDATION
# ============================================================

def validate_required_files():
    """
    Verify that all required ML assets exist.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Face-shape model not found:\n{MODEL_PATH}"
        )

    if not LANDMARK_MODEL_PATH.exists():
        raise FileNotFoundError(
            "MediaPipe Face Landmarker model not found:\n"
            f"{LANDMARK_MODEL_PATH}"
        )


# ============================================================
# MEDIAPIPE
# ============================================================

def detect_landmarks(image):
    """
    Detect MediaPipe Face Landmarks.

    A fresh MediaPipe FaceLandmarker instance is created
    for each image. This keeps the function self-contained
    and avoids relying on an undefined global LANDMARKER.

    Returns
    -------
    list
        The 478 normalized landmarks for the first face.
    """

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image,
    )

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

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(
                LANDMARK_MODEL_PATH
            )
        ),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with FaceLandmarker.create_from_options(
        options
    ) as landmarker:

        result = landmarker.detect(
            mp_image
        )

    if not result.face_landmarks:
        return None

    return result.face_landmarks[0]


# ============================================================
# EYE GEOMETRY
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

    return (
        landmark.x * image_width,
        landmark.y * image_height,
    )


def get_eye_geometry(
    landmarks,
    image_width,
    image_height,
):
    """
    Calculate eye centers, inter-eye distance
    and face rotation angle.

    These values are used by the virtual try-on
    system to position and rotate eyewear.
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
        (left_outer[0] + left_inner[0]) / 2,
        (left_outer[1] + left_inner[1]) / 2,
    )

    right_eye_center = (
        (right_outer[0] + right_inner[0]) / 2,
        (right_outer[1] + right_inner[1]) / 2,
    )

    eye_dx = (
        right_eye_center[0]
        - left_eye_center[0]
    )

    eye_dy = (
        right_eye_center[1]
        - left_eye_center[1]
    )

    eye_distance = (
        eye_dx ** 2
        + eye_dy ** 2
    ) ** 0.5

    import math

    angle = math.degrees(
        math.atan2(
            eye_dy,
            eye_dx,
        )
    )

    eye_center = (
        (
            left_eye_center[0]
            + right_eye_center[0]
        )
        / 2,
        (
            left_eye_center[1]
            + right_eye_center[1]
        )
        / 2,
    )

    return {
        "left_eye_center": left_eye_center,
        "right_eye_center": right_eye_center,
        "eye_center": eye_center,
        "eye_distance": eye_distance,
        "angle": angle,
    }


# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_image(image_path):
    """
    Run the complete face-analysis pipeline.

    This function is used by:

        1. Streamlit
        2. Command line
        3. Virtual Try-On

    Returns
    -------
    dict
        Structured face analysis containing:

        - image
        - predicted_face_shape
        - confidence
        - probabilities
        - features
        - landmark_count
        - landmarks
        - eye_geometry
    """

    validate_required_files()

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise ValueError(
            f"Could not read image:\n{image_path}"
        )

    image_height, image_width = image.shape[:2]

    # --------------------------------------------------------
    # Detect face landmarks
    # --------------------------------------------------------

    landmarks = detect_landmarks(
        image
    )

    if landmarks is None:
        raise ValueError(
            "No face detected. "
            "Please upload a clear front-facing photo."
        )

    landmark_count = len(
        landmarks
    )

    # --------------------------------------------------------
    # Extract facial geometry
    # --------------------------------------------------------

    features = extract_features(
        landmarks
    )

    # --------------------------------------------------------
    # Calculate eye geometry
    # --------------------------------------------------------

    eye_geometry = get_eye_geometry(
        landmarks,
        image_width,
        image_height,
    )

    # --------------------------------------------------------
    # Prepare ML features
    # --------------------------------------------------------

    feature_df = pd.DataFrame(
        [features],
        columns=FEATURE_COLUMNS,
    )

    # --------------------------------------------------------
    # Load trained face-shape model
    # --------------------------------------------------------

    saved_model = joblib.load(
        MODEL_PATH
    )

    # Your saved model is stored as:
    #
    # {
    #     "model": trained_pipeline,
    #     ...
    # }
    #
    # Use the stored pipeline.

    if isinstance(saved_model, dict):
        if "model" not in saved_model:
            raise ValueError(
                "Saved face-shape model does not contain "
                "the expected 'model' key."
            )

        model = saved_model["model"]

    else:
        # Supports a directly serialized sklearn pipeline.
        model = saved_model

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        feature_df
    )[0]

    probabilities = model.predict_proba(
        feature_df
    )[0]

    classes = model.classes_

    probability_df = pd.DataFrame(
        {
            "face_shape": classes,
            "probability": probabilities,
        }
    )

    probability_df = probability_df.sort_values(
        "probability",
        ascending=False,
    ).reset_index(drop=True)

    confidence = float(
        probability_df.iloc[0]["probability"]
    )

    # --------------------------------------------------------
    # Probability dictionary
    # --------------------------------------------------------

    probability_dict = {
        str(row["face_shape"]): float(
            row["probability"]
        )
        for _, row in probability_df.iterrows()
    }

    # --------------------------------------------------------
    # Return structured result
    # --------------------------------------------------------

    return {
        "image": str(image_path),
        "predicted_face_shape": str(
            prediction
        ),
        "confidence": confidence,
        "probabilities": probability_dict,
        "features": features,
        "landmark_count": landmark_count,
        "landmarks": landmarks,
        "eye_geometry": eye_geometry,
    }


# ============================================================
# SAVE ANALYSIS
# ============================================================

def save_analysis(
    result,
    output_path=OUTPUT_PATH,
):
    """
    Save face analysis results to CSV.

    Landmark objects are intentionally NOT saved because
    they are runtime objects used by the virtual try-on
    pipeline.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "image": result["image"],
        "predicted_face_shape": (
            result["predicted_face_shape"]
        ),
        "confidence": result["confidence"],
        "landmark_count": result[
            "landmark_count"
        ],
    }

    # --------------------------------------------------------
    # Save probabilities
    # --------------------------------------------------------

    for (
        face_shape,
        probability,
    ) in result[
        "probabilities"
    ].items():

        column = (
            "probability_"
            + face_shape.lower()
        )

        output[column] = probability

    # --------------------------------------------------------
    # Save geometry
    # --------------------------------------------------------

    output.update(
        result["features"]
    )

    # --------------------------------------------------------
    # Save eye geometry
    # --------------------------------------------------------

    eye_geometry = result[
        "eye_geometry"
    ]

    output["left_eye_x"] = (
        eye_geometry["left_eye_center"][0]
    )

    output["left_eye_y"] = (
        eye_geometry["left_eye_center"][1]
    )

    output["right_eye_x"] = (
        eye_geometry["right_eye_center"][0]
    )

    output["right_eye_y"] = (
        eye_geometry["right_eye_center"][1]
    )

    output["eye_distance"] = (
        eye_geometry["eye_distance"]
    )

    output["face_rotation_angle"] = (
        eye_geometry["angle"]
    )

    # --------------------------------------------------------
    # Write CSV
    # --------------------------------------------------------

    pd.DataFrame(
        [output]
    ).to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    return output_path


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():

    print("=" * 70)
    print(
        "AURA — REAL USER FACE ANALYZER"
    )
    print("=" * 70)

    try:

        result = analyze_image(
            DEFAULT_IMAGE_PATH
        )

    except Exception as error:

        print(
            f"\nERROR: {error}"
        )

        return

    # --------------------------------------------------------
    # Detection
    # --------------------------------------------------------

    print(
        "\nFace detected."
    )

    print(
        f"Landmarks detected: "
        f"{result['landmark_count']}"
    )

    # --------------------------------------------------------
    # Eye geometry
    # --------------------------------------------------------

    eye_geometry = result[
        "eye_geometry"
    ]

    print(
        "\nVirtual Try-On geometry:"
    )

    print(
        "Left eye center: "
        f"{eye_geometry['left_eye_center']}"
    )

    print(
        "Right eye center: "
        f"{eye_geometry['right_eye_center']}"
    )

    print(
        "Eye distance: "
        f"{eye_geometry['eye_distance']:.2f}px"
    )

    print(
        "Face rotation angle: "
        f"{eye_geometry['angle']:.2f}°"
    )

    # --------------------------------------------------------
    # Facial geometry
    # --------------------------------------------------------

    print(
        "\nFacial features:"
    )

    for column in FEATURE_COLUMNS:

        print(
            f"{column:30s}: "
            f"{result['features'][column]:.6f}"
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "FACE SHAPE PREDICTION"
    )

    print(
        "=" * 70
    )

    print(
        f"\nPredicted face shape: "
        f"{result['predicted_face_shape']}"
    )

    print(
        f"Top probability: "
        f"{result['confidence']:.4f}"
    )

    # --------------------------------------------------------
    # Probability distribution
    # --------------------------------------------------------

    print(
        "\nFace-shape probabilities:"
    )

    sorted_probabilities = sorted(
        result["probabilities"].items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for (
        face_shape,
        probability,
    ) in sorted_probabilities:

        print(
            f"{face_shape:10s}: "
            f"{probability:.4f}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    saved_path = save_analysis(
        result
    )

    print(
        "\nSaved analysis:"
    )

    print(
        saved_path.resolve()
    )

    print(
        "\nFace analysis completed."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
from pathlib import Path

from PIL import Image

from src.analyze_user_face import analyze_image
from src.virtual_try_on import apply_virtual_try_on


PROJECT_ROOT = Path(__file__).resolve().parents[1]

USER_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "face_landmarks_test.jpg"
)

FRAME_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "catalogue"
    / "images"
    / "FRAME_001.jpg"
)

OUTPUT_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "virtual_tryon_FRAME_001.jpg"
)


def main():

    print("=" * 70)
    print("AURA — VIRTUAL TRY-ON V1")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    if not USER_IMAGE.exists():
        raise FileNotFoundError(
            f"User image not found:\n{USER_IMAGE}"
        )

    if not FRAME_IMAGE.exists():
        raise FileNotFoundError(
            f"Frame image not found:\n{FRAME_IMAGE}"
        )

    # --------------------------------------------------------
    # Analyze face
    # --------------------------------------------------------

    print("\nAnalyzing face...")

    analysis = analyze_image(
        USER_IMAGE
    )

    print(
        f"Face shape: "
        f"{analysis['predicted_face_shape']}"
    )

    print(
        f"Landmarks: "
        f"{analysis['landmark_count']}"
    )

    # --------------------------------------------------------
    # Load user image
    # --------------------------------------------------------

    user_image = Image.open(
        USER_IMAGE
    ).convert("RGB")

    # --------------------------------------------------------
    # Apply virtual try-on
    # --------------------------------------------------------

    print("\nApplying FRAME_001...")

    result, geometry = apply_virtual_try_on(
        image=user_image,
        landmarks=analysis["landmarks"],
        frame_path=FRAME_IMAGE,
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    result.save(
        OUTPUT_IMAGE,
        quality=95,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("\nVirtual Try-On completed.")

    print(
        f"Eye distance: "
        f"{geometry['eye_distance']:.2f}px"
    )

    print(
        f"Face rotation: "
        f"{geometry['angle']:.2f}°"
    )

    print(
        f"\nOutput:"
    )

    print(
        OUTPUT_IMAGE.resolve()
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
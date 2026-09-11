from pathlib import Path
import sys

import pandas as pd
import streamlit as st
from PIL import Image
import streamlit_shadcn_ui as ui


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analyze_user_face import analyze_image
from src.recommend_from_real_face import recommend_from_face_analysis
from src.personalization_loop import record_feedback
from src.ai_stylist import generate_stylist_recommendations
from src.virtual_try_on import apply_virtual_try_on


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AURA — AI Eyewear",
    page_icon="👓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL STYLING
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background: #f7f7f5;
        }

        .block-container {
            max-width: 1450px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        .aura-title {
            font-size: 54px;
            font-weight: 800;
            line-height: 1;
            letter-spacing: -0.055em;
            margin-top: 10px;
            margin-bottom: 15px;
        }

        .aura-subtitle {
            max-width: 720px;
            color: #6b6b68;
            font-size: 17px;
            line-height: 1.6;
            margin-bottom: 30px;
        }

        .section-label {
            font-size: 11px;
            font-weight: 700;
            color: #8a8a85;
            letter-spacing: 0.15em;
            text-transform: uppercase;
        }

        .stylist-box {
            background: #efefec;
            border-radius: 10px;
            padding: 14px;
            margin-top: 10px;
            margin-bottom: 12px;
        }

        .stylist-rating {
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #666660;
            margin-bottom: 7px;
        }

        .stylist-title {
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 6px;
        }

        .stylist-text {
            color: #5f5f5b;
            font-size: 13px;
            line-height: 1.55;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

USER_ID = "USER_002"

CATALOGUE_PATH = (
    PROJECT_ROOT
    / "data"
    / "catalogue"
    / "eyewear_catalogue.csv"
)


# ============================================================
# SESSION STATE
# ============================================================

if "face_analysis" not in st.session_state:
    st.session_state.face_analysis = None

if "uploaded_image_name" not in st.session_state:
    st.session_state.uploaded_image_name = None

if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

if "stylist_recommendations" not in st.session_state:
    st.session_state.stylist_recommendations = None

if "feedback_message" not in st.session_state:
    st.session_state.feedback_message = None

if "user_image_path" not in st.session_state:
    st.session_state.user_image_path = None

if "try_on_image" not in st.session_state:
    st.session_state.try_on_image = None

if "try_on_frame" not in st.session_state:
    st.session_state.try_on_frame = None

if "try_on_geometry" not in st.session_state:
    st.session_state.try_on_geometry = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def build_user_preferences(style, color, frame_type, budget):
    """
    Build the preference dictionary consumed by the
    recommendation engine.
    """

    return {
        "preferred_style": style,
        "preferred_color": color,
        "preferred_frame_type": frame_type,
        "budget": budget,
    }


def generate_stylist_data(
    recommendations,
    face_result,
    style,
    color,
    frame_type,
    budget,
):
    """
    Generate AI Stylist explanations directly from the
    current recommendation results.
    """

    if recommendations is None:
        return None

    if recommendations.empty:
        return None

    catalogue = pd.read_csv(
        CATALOGUE_PATH
    )

    user = {
        "user_id": USER_ID,
        "face_shape": str(
            face_result["predicted_face_shape"]
        ),
        "preferred_style": style,
        "preferred_color": color,
        "preferred_frame_type": frame_type,
        "budget": budget,
    }

    stylist_data = (
        generate_stylist_recommendations(
            recommendations,
            user,
            catalogue,
            USER_ID,
        )
    )

    return stylist_data


def regenerate_recommendations(
    face_result,
    style,
    color,
    frame_type,
    budget,
):
    """
    Re-run the recommendation engine after feedback or
    preference changes.
    """

    preferences = build_user_preferences(
        style,
        color,
        frame_type,
        budget,
    )

    recommendations = (
        recommend_from_face_analysis(
            face_result,
            preferences,
        )
    )

    stylist_data = generate_stylist_data(
        recommendations,
        face_result,
        style,
        color,
        frame_type,
        budget,
    )

    st.session_state.recommendations = (
        recommendations
    )

    st.session_state.stylist_recommendations = (
        stylist_data
    )


def generate_try_on_image(
    user_image_path,
    frame_image_path,
    landmarks,
):
    """
    Generate a V1 virtual try-on preview using
    MediaPipe eye geometry and the selected frame.
    """

    user_image = Image.open(
        user_image_path
    ).convert("RGB")

    result, geometry = apply_virtual_try_on(
        image=user_image,
        landmarks=landmarks,
        frame_path=frame_image_path,
    )

    return result, geometry


def handle_try_on(
    frame_id,
    frame_image_name,
    face_result,
):
    """
    Generate and store a virtual try-on preview.
    """

    user_image_path = st.session_state.get(
        "user_image_path"
    )

    if not user_image_path:
        st.warning(
            "Please analyse a photo before using Virtual Try-On."
        )
        return

    landmarks = face_result.get("landmarks") if face_result else None

    if not landmarks:
        st.warning(
            "Face landmarks are unavailable. Please analyse your photo again."
        )
        return

    frame_path = (
        PROJECT_ROOT
        / "data"
        / "catalogue"
        / "images"
        / str(frame_image_name)
    )

    if not frame_path.exists():
        st.error(
            f"Frame image not found: {frame_path}"
        )
        return

    try:
        result, geometry = generate_try_on_image(
            user_image_path,
            frame_path,
            landmarks,
        )

        st.session_state.try_on_image = result
        st.session_state.try_on_frame = str(frame_id)
        st.session_state.try_on_geometry = geometry

    except Exception as error:
        st.error(
            f"Virtual Try-On failed: {error}"
        )


def handle_feedback(
    frame_id,
    liked,
    face_result,
    style,
    color,
    frame_type,
    budget,
):
    """
    Save feedback, rebuild personalization and immediately
    regenerate recommendations.
    """

    try:

        record_feedback(
            user_id=USER_ID,
            frame_id=frame_id,
            liked=liked,
        )

        feedback_text = (
            "liked"
            if liked
            else "disliked"
        )

        st.session_state.feedback_message = (
            f"Feedback saved — {frame_id} "
            f"marked as {feedback_text}."
        )

        regenerate_recommendations(
            face_result,
            style,
            color,
            frame_type,
            budget,
        )

        st.rerun()

    except Exception as error:

        st.error(
            f"Could not save feedback: {error}"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("AURA")

    st.caption(
        "AI EYEWEAR INTELLIGENCE"
    )

    st.divider()

    st.subheader(
        "Style profile"
    )

    style = st.selectbox(
        "Preferred style",
        [
            "Casual",
            "Fashion",
            "Professional",
        ],
    )

    color = st.selectbox(
        "Preferred colour",
        [
            "Black",
            "Gold",
            "Silver",
            "Transparent",
            "Purple",
        ],
    )

    frame_type = st.selectbox(
        "Frame type",
        [
            "Optical",
            "Sunglasses",
        ],
    )

    budget = st.number_input(
        "Maximum budget",
        min_value=500,
        max_value=100000,
        value=3000,
        step=500,
        format="%d",
    )

    st.divider()

    st.subheader(
        "AI pipeline"
    )

    st.checkbox(
        "Face detection",
        value=True,
        disabled=True,
    )

    st.checkbox(
        "Facial geometry",
        value=True,
        disabled=True,
    )

    st.checkbox(
        "Face classification",
        value=True,
        disabled=True,
    )

    st.checkbox(
        "Visual embeddings",
        value=True,
        disabled=True,
    )

    st.checkbox(
        "Hybrid ranking",
        value=True,
        disabled=True,
    )

    st.checkbox(
        "Virtual Try-On",
        value=True,
        disabled=True,
    )

    st.divider()

    st.caption(
        "Catalogue: 7 frames"
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    '<div class="section-label">'
    'AURA / INTELLIGENT EYEWEAR'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="aura-title">
        Find the frame<br>
        that fits you.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="aura-subtitle">
        A personalized eyewear recommendation system combining
        facial geometry, visual similarity and individual style
        preferences.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MAIN TABS
# ============================================================

tab_analysis, tab_preferences, tab_about = st.tabs(
    [
        "Face analysis",
        "Preferences",
        "How it works",
    ]
)


# ============================================================
# FACE ANALYSIS TAB
# ============================================================

with tab_analysis:

    st.subheader(
        "Upload your photo"
    )

    st.caption(
        "Use a clear, front-facing photograph for the best "
        "facial geometry analysis."
    )

    upload_col, preview_col = st.columns(
        [1, 1],
        gap="large",
    )

    # ========================================================
    # UPLOAD
    # ========================================================

    with upload_col:

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=[
                "jpg",
                "jpeg",
                "png",
            ],
        )

        if uploaded_file:

            st.success(
                f"Image loaded: "
                f"{uploaded_file.name}"
            )

            if (
                st.session_state.uploaded_image_name
                != uploaded_file.name
            ):

                st.session_state.face_analysis = None

                st.session_state.recommendations = None

                st.session_state.stylist_recommendations = None

                st.session_state.feedback_message = None
                st.session_state.user_image_path = None
                st.session_state.try_on_image = None
                st.session_state.try_on_frame = None
                st.session_state.try_on_geometry = None

                st.session_state.uploaded_image_name = (
                    uploaded_file.name
                )

    # ========================================================
    # PREVIEW
    # ========================================================

    with preview_col:

        if uploaded_file:

            st.image(
                uploaded_file,
                caption="Photo preview",
                width=400,
            )

        else:

            st.info(
                "Your photo preview will appear here."
            )

    st.divider()

    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    analyze_button = st.button(
        "Analyse photo",
        type="primary",
        width="stretch",
    )

    if analyze_button:

        if uploaded_file is None:

            st.warning(
                "Please upload a photo first."
            )

        else:

            try:

                # ------------------------------------------------
                # Save uploaded image for the current session
                # ------------------------------------------------

                suffix = (
                    Path(uploaded_file.name).suffix.lower()
                    or ".jpg"
                )

                upload_directory = (
                    PROJECT_ROOT
                    / "data"
                    / "processed"
                    / "faces"
                    / "user_uploads"
                )

                upload_directory.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                user_image_path = (
                    upload_directory
                    / f"{USER_ID}_current{suffix}"
                )

                user_image_path.write_bytes(
                    uploaded_file.getbuffer()
                )

                st.session_state.user_image_path = str(
                    user_image_path
                )

                # ------------------------------------------------
                # Face analysis
                # ------------------------------------------------

                with st.spinner(
                    "Analysing facial geometry..."
                ):

                    result = analyze_image(
                        user_image_path
                    )

                st.session_state.face_analysis = (
                    result
                )

                # ------------------------------------------------
                # Generate recommendations
                # ------------------------------------------------

                with st.spinner(
                    "Generating personalized frame "
                    "recommendations..."
                ):

                    regenerate_recommendations(
                        result,
                        style,
                        color,
                        frame_type,
                        budget,
                    )

                st.session_state.feedback_message = None

                st.success(
                    "Face analysis and recommendations "
                    "completed."
                )

            except Exception as error:

                st.error(
                    f"Analysis failed: {error}"
                )

    # ========================================================
    # RESULTS
    # ========================================================

    if st.session_state.face_analysis:

        result = (
            st.session_state.face_analysis
        )

        # ====================================================
        # FACE PROFILE
        # ====================================================

        st.divider()

        st.subheader(
            "Your face profile"
        )

        result_col, probability_col = st.columns(
            [0.8, 1.2],
            gap="large",
        )

        # ----------------------------------------------------
        # Main prediction
        # ----------------------------------------------------

        with result_col:

            st.markdown(
                "#### Predicted face shape"
            )

            ui.metric_card(
                "Primary prediction",
                result[
                    "predicted_face_shape"
                ],
                description=(
                    f"Model probability: "
                    f"{result['confidence'] * 100:.1f}%"
                ),
                key="face_shape_result",
            )

            st.write("")

            c1, c2 = st.columns(2)

            with c1:

                ui.metric_card(
                    "Landmarks",
                    str(
                        result[
                            "landmark_count"
                        ]
                    ),
                    description="MediaPipe",
                    key="landmark_result",
                )

            with c2:

                ui.metric_card(
                    "Classifier",
                    "SVM",
                    description="Geometry model",
                    key="classifier_result",
                )

        # ----------------------------------------------------
        # Probability distribution
        # ----------------------------------------------------

        with probability_col:

            st.markdown(
                "#### Prediction distribution"
            )

            st.caption(
                "The model retains uncertainty across all "
                "five face-shape categories."
            )

            probability_data = []

            for (
                face_shape,
                probability,
            ) in result[
                "probabilities"
            ].items():

                probability_data.append(
                    {
                        "Face shape": face_shape,
                        "Probability": round(
                            probability * 100,
                            2,
                        ),
                    }
                )

            probability_df = pd.DataFrame(
                probability_data
            )

            probability_df = (
                probability_df
                .sort_values(
                    "Probability",
                    ascending=False,
                )
                .reset_index(
                    drop=True
                )
            )

            st.dataframe(
                probability_df,
                width="stretch",
                hide_index=True,
            )

            st.bar_chart(
                probability_df.set_index(
                    "Face shape"
                ),
                width="stretch",
            )

        # ====================================================
        # FACIAL GEOMETRY
        # ====================================================

        st.divider()

        st.subheader(
            "Facial geometry"
        )

        st.caption(
            "The following features are the same features "
            "used during model training."
        )

        features = result[
            "features"
        ]

        geometry = {
            "Face width": features[
                "face_width"
            ],
            "Face height": features[
                "face_height"
            ],
            "Aspect ratio": features[
                "face_aspect_ratio"
            ],
            "Forehead width": features[
                "forehead_width"
            ],
            "Cheek width": features[
                "cheek_width"
            ],
            "Jaw width": features[
                "jaw_width"
            ],
            "Forehead / face": features[
                "forehead_to_face_ratio"
            ],
            "Cheek / face": features[
                "cheek_to_face_ratio"
            ],
            "Jaw / face": features[
                "jaw_to_face_ratio"
            ],
        }

        geometry_df = pd.DataFrame(
            [
                {
                    "Feature": name,
                    "Value": round(
                        value,
                        6,
                    ),
                }
                for name, value in geometry.items()
            ]
        )

        st.dataframe(
            geometry_df,
            width="stretch",
            hide_index=True,
        )

        # ====================================================
        # RECOMMENDATIONS
        # ====================================================

        st.divider()

        st.subheader(
            "Recommended frames"
        )

        st.caption(
            "Ranked using facial compatibility, style "
            "preferences, visual preference signals and "
            "the hybrid recommendation model."
        )

        # ----------------------------------------------------
        # Feedback status
        # ----------------------------------------------------

        if st.session_state.feedback_message:

            st.success(
                st.session_state.feedback_message
            )

        recommendations = (
            st.session_state.recommendations
        )

        stylist_recommendations = (
            st.session_state.stylist_recommendations
        )

        if recommendations is None:

            st.info(
                "Run the analysis to generate "
                "personalized frame recommendations."
            )

        elif recommendations.empty:

            st.warning(
                "No frames matched the current preferences "
                "and budget."
            )

        else:

            # =================================================
            # RECOMMENDATION CARDS
            # =================================================

            display_count = min(
                5,
                len(recommendations),
            )

            recommendation_columns = st.columns(
                display_count,
                gap="medium",
            )

            for index in range(
                display_count
            ):

                row = recommendations.iloc[
                    index
                ]

                frame_id = str(
                    row["frame_id"]
                )

                with recommendation_columns[
                    index
                ]:

                    # =========================================
                    # IMAGE
                    # =========================================

                    frame_image = (
                        PROJECT_ROOT
                        / "data"
                        / "catalogue"
                        / "images"
                        / str(
                            row["frame_image"]
                        )
                    )

                    if frame_image.exists():

                        st.image(
                            str(frame_image),
                            width="stretch",
                        )

                    else:

                        st.info(
                            "Frame image unavailable."
                        )

                    # =========================================
                    # FRAME DETAILS
                    # =========================================

                    st.markdown(
                        f"### #{int(row['rank'])}"
                    )

                    st.markdown(
                        f"**{row['brand']}**"
                    )

                    st.caption(
                        f"{row['frame_shape']} · "
                        f"{row['frame_color']} · "
                        f"{row['style']}"
                    )

                    ui.metric_card(
                        "Hybrid score",
                        (
                            f"{float(row['hybrid_score']) * 100:.1f}%"
                        ),
                        description=(
                            "Recommendation score"
                        ),
                        key=(
                            f"recommendation_score_"
                            f"{frame_id}"
                        ),
                    )

                    # =========================================
                    # AI STYLIST
                    # =========================================

                    if (
                        stylist_recommendations
                        is not None
                    ):

                        stylist_rows = (
                            stylist_recommendations[
                                stylist_recommendations[
                                    "frame_id"
                                ].astype(str)
                                == frame_id
                            ]
                        )

                        if not stylist_rows.empty:

                            stylist_row = (
                                stylist_rows.iloc[0]
                            )

                            st.markdown(
                                """
                                <div class="stylist-box">
                                    <div class="stylist-title">
                                        Why this frame?
                                    </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            st.markdown(
                                (
                                    '<div class="stylist-rating">'
                                    f"{stylist_row['stylist_rating']}"
                                    "</div>"
                                ),
                                unsafe_allow_html=True,
                            )

                            st.markdown(
                                (
                                    '<div class="stylist-text">'
                                    f"{stylist_row['explanation']}"
                                    "</div></div>"
                                ),
                                unsafe_allow_html=True,
                            )

                    st.write("")

                    # =========================================
                    # VIRTUAL TRY-ON
                    # =========================================

                    if st.button(
                        "Try On",
                        key=f"try_on_{frame_id}",
                        width="stretch",
                    ):
                        handle_try_on(
                            frame_id=frame_id,
                            frame_image_name=row["frame_image"],
                            face_result=result,
                        )
                        st.rerun()

                    # =========================================
                    # FEEDBACK
                    # =========================================

                    feedback_col1, feedback_col2 = (
                        st.columns(2)
                    )

                    with feedback_col1:

                        if st.button(
                            "Like",
                            key=f"like_{frame_id}",
                            width="stretch",
                        ):

                            handle_feedback(
                                frame_id=frame_id,
                                liked=True,
                                face_result=result,
                                style=style,
                                color=color,
                                frame_type=frame_type,
                                budget=budget,
                            )

                    with feedback_col2:

                        if st.button(
                            "Dislike",
                            key=f"dislike_{frame_id}",
                            width="stretch",
                        ):

                            handle_feedback(
                                frame_id=frame_id,
                                liked=False,
                                face_result=result,
                                style=style,
                                color=color,
                                frame_type=frame_type,
                                budget=budget,
                            )

            # =================================================
            # VIRTUAL TRY-ON PREVIEW
            # =================================================

            if st.session_state.try_on_image is not None:

                st.divider()

                st.subheader(
                    "Virtual Try-On"
                )

                st.caption(
                    "V1 landmark-based preview using the selected frame."
                )

                try_on_col1, try_on_col2 = st.columns(
                    [1.25, 0.75],
                    gap="large",
                )

                with try_on_col1:

                    st.image(
                        st.session_state.try_on_image,
                        width="stretch",
                    )

                with try_on_col2:

                    st.markdown(
                        "#### Selected frame"
                    )

                    st.markdown(
                        f"**{st.session_state.try_on_frame}**"
                    )

                    try_on_geometry = (
                        st.session_state.try_on_geometry
                    )

                    if try_on_geometry:

                        ui.metric_card(
                            "Eye distance",
                            f"{try_on_geometry['eye_distance']:.1f} px",
                            description="Detected inter-eye distance",
                            key="try_on_eye_distance",
                        )

                        ui.metric_card(
                            "Face rotation",
                            f"{try_on_geometry['angle']:.1f}°",
                            description="Estimated eye-line rotation",
                            key="try_on_angle",
                        )

                    st.caption(
                        "The preview is a computer-vision overlay and "
                        "is not a measurement of actual frame fit."
                    )

                    if st.button(
                        "Close preview",
                        key="close_try_on",
                        width="stretch",
                    ):
                        st.session_state.try_on_image = None
                        st.session_state.try_on_frame = None
                        st.session_state.try_on_geometry = None
                        st.rerun()

            # =================================================
            # RECOMMENDATION SIGNALS
            # =================================================

            st.divider()

            st.markdown(
                "#### Recommendation signals"
            )

            signal_columns = st.columns(3)

            top = recommendations.iloc[0]

            with signal_columns[0]:

                ui.metric_card(
                    "Face compatibility",
                    (
                        f"{float(top['face_compatibility']) * 100:.1f}%"
                    ),
                    description=(
                        "Uncertainty-aware facial fit"
                    ),
                    key="top_face_compatibility",
                )

            with signal_columns[1]:

                ui.metric_card(
                    "Predicted preference",
                    (
                        f"{float(top['model_probability']) * 100:.1f}%"
                    ),
                    description=(
                        "Trained recommendation-model signal"
                    ),
                    key="top_model_probability",
                )

            with signal_columns[2]:

                ui.metric_card(
                    "Visual preference",
                    (
                        f"{float(top['visual_preference_score']) * 100:.1f}%"
                    ),
                    description=(
                        "Similarity to positive feedback"
                    ),
                    key="top_visual_preference",
                )

            st.caption(
                "Scores are model signals, not guarantees "
                "that a user will like or purchase a frame."
            )

            # =================================================
            # PERSONALIZATION
            # =================================================

            st.divider()

            st.markdown(
                "#### Personalization"
            )

            feedback_path = (
                PROJECT_ROOT
                / "data"
                / "raw"
                / "recommendations"
                / "feedback.csv"
            )

            if feedback_path.exists():

                try:

                    feedback_df = pd.read_csv(
                        feedback_path
                    )

                    user_feedback = feedback_df[
                        feedback_df[
                            "user_id"
                        ].astype(str)
                        == USER_ID
                    ]

                    liked_frames = (
                        user_feedback[
                            user_feedback[
                                "liked"
                            ] == 1
                        ]["frame_id"]
                        .astype(str)
                        .tolist()
                    )

                    disliked_frames = (
                        user_feedback[
                            user_feedback[
                                "liked"
                            ] == 0
                        ]["frame_id"]
                        .astype(str)
                        .tolist()
                    )

                    personalization_col1, personalization_col2 = (
                        st.columns(2)
                    )

                    with personalization_col1:

                        ui.metric_card(
                            "Liked frames",
                            str(
                                len(liked_frames)
                            ),
                            description=(
                                "Positive feedback signals"
                            ),
                            key="liked_frames_count",
                        )

                    with personalization_col2:

                        ui.metric_card(
                            "Disliked frames",
                            str(
                                len(disliked_frames)
                            ),
                            description=(
                                "Negative feedback signals"
                            ),
                            key="disliked_frames_count",
                        )

                    if liked_frames:

                        st.caption(
                            "Liked: "
                            + ", ".join(
                                liked_frames
                            )
                        )

                    if disliked_frames:

                        st.caption(
                            "Disliked: "
                            + ", ".join(
                                disliked_frames
                            )
                        )

                    st.info(
                        "Your feedback updates the visual "
                        "preference profile and influences "
                        "future recommendations."
                    )

                except Exception as error:

                    st.caption(
                        "Personalization data unavailable: "
                        f"{error}"
                    )

            # =================================================
            # FULL DATA
            # =================================================

            with st.expander(
                "View full recommendation data"
            ):

                st.dataframe(
                    recommendations[
                        [
                            "rank",
                            "frame_id",
                            "brand",
                            "frame_shape",
                            "frame_color",
                            "style",
                            "face_compatibility",
                            "model_probability",
                            "visual_preference_score",
                            "hybrid_score",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )


# ============================================================
# PREFERENCES TAB
# ============================================================

with tab_preferences:

    st.subheader(
        "Your style profile"
    )

    st.caption(
        "These values are passed directly to the "
        "recommendation engine."
    )

    c1, c2 = st.columns(2)

    with c1:

        ui.metric_card(
            "Preferred style",
            style,
            description="Aesthetic preference",
            key="preference_style",
        )

        ui.metric_card(
            "Frame type",
            frame_type,
            description="Product category",
            key="preference_frame_type",
        )

    with c2:

        ui.metric_card(
            "Preferred colour",
            color,
            description="Frame colour",
            key="preference_color",
        )

        ui.metric_card(
            "Maximum budget",
            f"₹{budget:,}",
            description="Price ceiling",
            key="preference_budget",
        )


# ============================================================
# HOW IT WORKS
# ============================================================

with tab_about:

    st.subheader(
        "How AURA works"
    )

    st.write(
        "AURA is designed as a hybrid recommendation system "
        "rather than a simple face-shape lookup."
    )

    st.divider()

    steps = [
        (
            "01",
            "Face analysis",
            "MediaPipe detects facial landmarks and extracts "
            "facial geometry.",
        ),
        (
            "02",
            "Face classification",
            "A trained SVM estimates the probability distribution "
            "across five face-shape categories.",
        ),
        (
            "03",
            "Visual understanding",
            "ResNet-50 embeddings represent visual characteristics "
            "of faces and eyewear.",
        ),
        (
            "04",
            "Hybrid ranking",
            "Metadata compatibility and visual preference are "
            "combined to rank candidate frames.",
        ),
        (
            "05",
            "Personalization",
            "User feedback updates the visual preference profile "
            "for future recommendations.",
        ),
        (
            "06",
            "AI Stylist",
            "A stylist explanation engine converts recommendation "
            "signals into understandable reasons for each frame.",
        ),
        (
            "07",
            "Virtual Try-On",
            "MediaPipe eye geometry aligns the selected frame to "
            "the user's face using scale, position and rotation.",
        ),
    ]

    for number, title, description in steps:

        with st.container(
            border=True
        ):

            left, right = st.columns(
                [0.12, 0.88]
            )

            with left:

                st.markdown(
                    f"### {number}"
                )

            with right:

                st.markdown(
                    f"**{title}**"
                )

                st.caption(
                    description
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AURA · AI Eyewear Recommendation & Virtual Styling System"
)
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# Files
# --------------------------------------------------

CATALOGUE_FILE = "data/catalogue/eyewear_catalogue.csv"

EMBEDDINGS_FILE = (
    "data/processed/eyewear/frame_embeddings.npy"
)

METADATA_FILE = (
    "data/processed/eyewear/frame_metadata.csv"
)


# --------------------------------------------------
# Baseline face → frame compatibility
# --------------------------------------------------

COMPATIBILITY_RULES = {

    "Heart": {
        "Aviator": 1.00,
        "Round": 0.90,
        "Cat Eye": 0.80,
        "Rectangle": 0.70,
        "Hexagon": 0.60,
        "Pentagon": 0.60,
    },

    "Oblong": {
        "Round": 1.00,
        "Aviator": 0.90,
        "Cat Eye": 0.80,
        "Rectangle": 0.70,
        "Hexagon": 0.70,
        "Pentagon": 0.70,
    },

    "Oval": {
        "Rectangle": 1.00,
        "Square": 0.90,
        "Aviator": 0.90,
        "Round": 0.80,
        "Hexagon": 0.80,
        "Pentagon": 0.80,
        "Cat Eye": 0.80,
    },

    "Round": {
        "Rectangle": 1.00,
        "Hexagon": 0.90,
        "Pentagon": 0.85,
        "Aviator": 0.80,
        "Square": 0.80,
        "Cat Eye": 0.70,
        "Round": 0.60,
    },

    "Square": {
        "Round": 1.00,
        "Aviator": 0.90,
        "Cat Eye": 0.80,
        "Hexagon": 0.80,
        "Rectangle": 0.70,
        "Pentagon": 0.70,
    }
}


# --------------------------------------------------
# Load data
# --------------------------------------------------

catalogue = pd.read_csv(CATALOGUE_FILE)

embeddings = np.load(EMBEDDINGS_FILE)

metadata = pd.read_csv(METADATA_FILE)


# --------------------------------------------------
# Verify alignment
# --------------------------------------------------

assert len(catalogue) == len(embeddings), (
    "Catalogue and embeddings have different sizes."
)

assert len(metadata) == len(embeddings), (
    "Metadata and embeddings have different sizes."
)

assert list(catalogue["frame_id"]) == list(metadata["frame_id"]), (
    "Catalogue and embedding metadata are not aligned."
)


# --------------------------------------------------
# Calculate visual centrality
# --------------------------------------------------
#
# With only 5 frames, we don't have a user-selected
# reference frame yet.
#
# Therefore we calculate how visually distinctive
# each frame is relative to the catalogue.
#
# This is NOT user preference.
#

similarity_matrix = cosine_similarity(embeddings)

visual_scores = similarity_matrix.mean(axis=1)

# Normalize to 0-1
visual_min = visual_scores.min()
visual_max = visual_scores.max()

if visual_max > visual_min:

    visual_scores = (
        (visual_scores - visual_min)
        / (visual_max - visual_min)
    )

else:

    visual_scores = np.ones(len(visual_scores))


# --------------------------------------------------
# Recommendation function
# --------------------------------------------------

def recommend_frames(
    face_shape,
    top_k=5,
    compatibility_weight=0.70,
    visual_weight=0.30
):

    face_shape = face_shape.strip().title()

    if face_shape not in COMPATIBILITY_RULES:

        raise ValueError(
            f"Unknown face shape: {face_shape}"
        )

    results = []

    for i, row in catalogue.iterrows():

        frame_shape = row["frame_shape"]

        # Face compatibility
        face_score = COMPATIBILITY_RULES[
            face_shape
        ].get(frame_shape, 0.50)

        # Visual score
        visual_score = float(
            visual_scores[i]
        )

        # Hybrid score
        final_score = (
            compatibility_weight * face_score
            +
            visual_weight * visual_score
        )

        results.append({

            "frame_id": row["frame_id"],

            "brand": row["brand"],

            "frame_shape": frame_shape,

            "frame_color": row["frame_color"],

            "face_compatibility": round(
                face_score, 3
            ),

            "visual_score": round(
                visual_score, 3
            ),

            "final_score": round(
                final_score, 3
            )
        })

    results = pd.DataFrame(results)

    results = results.sort_values(
        "final_score",
        ascending=False
    )

    return results.head(top_k)


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    for face_shape in [
        "Heart",
        "Oblong",
        "Oval",
        "Round",
        "Square"
    ]:

        print("\n" + "=" * 65)

        print(
            f"HYBRID RECOMMENDATIONS — "
            f"{face_shape} FACE"
        )

        print("=" * 65)

        recommendations = recommend_frames(
            face_shape,
            top_k=5
        )

        print(
            recommendations.to_string(
                index=False
            )
        )
import numpy as np
import pandas as pd

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path(
    "data/processed/recommendations/"
    "synthetic_hybrid_interactions.csv"
)

EMBEDDINGS_PATH = Path(
    "data/processed/eyewear/"
    "frame_embeddings.npy"
)

METADATA_PATH = Path(
    "data/processed/eyewear/"
    "frame_metadata.csv"
)

OUTPUT_PATH = Path(
    "data/processed/recommendations/"
    "hybrid_ranking_evaluation.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

embeddings = np.load(
    EMBEDDINGS_PATH
)

metadata = pd.read_csv(
    METADATA_PATH
)


print("\n==============================================")
print("HYBRID RECOMMENDER RANKING EVALUATION")
print("==============================================")


# ============================================================
# PREPARE EMBEDDING LOOKUP
# ============================================================

embedding_lookup = {}

for index, row in metadata.iterrows():

    embedding_lookup[
        str(row["frame_id"])
    ] = embeddings[index]


# ============================================================
# NORMALIZE EMBEDDINGS
# ============================================================

for frame_id in embedding_lookup:

    vector = embedding_lookup[frame_id]

    norm = np.linalg.norm(vector)

    if norm > 0:

        embedding_lookup[frame_id] = (
            vector / norm
        )


# ============================================================
# RECOMMENDATION FEATURES
# ============================================================

metadata_features = [
    "face_compatibility",
    "style_match",
    "color_match",
    "frame_type_match",
]


# ============================================================
# SIMPLE METADATA SCORE
# ============================================================

def metadata_score(row):

    return (
        0.50 * row["face_compatibility"]
        + 0.25 * row["style_match"]
        + 0.15 * row["color_match"]
        + 0.10 * row["frame_type_match"]
    )


# ============================================================
# NDCG
# ============================================================

def ndcg_at_k(
    ranked_relevance,
    k
):

    relevance = np.asarray(
        ranked_relevance[:k],
        dtype=float
    )

    if len(relevance) == 0:
        return 0.0

    discounts = np.log2(
        np.arange(
            2,
            len(relevance) + 2
        )
    )

    dcg = np.sum(
        (2 ** relevance - 1)
        / discounts
    )

    ideal_relevance = np.sort(
        relevance
    )[::-1]

    ideal_dcg = np.sum(
        (2 ** ideal_relevance - 1)
        / discounts
    )

    if ideal_dcg == 0:
        return 0.0

    return float(
        dcg / ideal_dcg
    )


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(
    relevance,
    k
):

    relevance = np.asarray(
        relevance,
        dtype=int
    )

    top_k = relevance[:k]

    relevant_total = np.sum(
        relevance
    )

    relevant_in_top_k = np.sum(
        top_k
    )

    precision = (
        relevant_in_top_k / k
    )

    if relevant_total > 0:

        recall = (
            relevant_in_top_k
            / relevant_total
        )

    else:

        recall = 0.0


    hit_rate = int(
        relevant_in_top_k > 0
    )


    ndcg = ndcg_at_k(
        relevance,
        k
    )


    return (
        precision,
        recall,
        hit_rate,
        ndcg
    )


# ============================================================
# EVALUATION
# ============================================================

results = []


users = df["user_id"].unique()


print(
    f"Users evaluated: {len(users)}"
)


for user_id in users:

    user_df = df[
        df["user_id"] == user_id
    ].copy()


    # --------------------------------------------------------
    # Need at least two liked frames.
    #
    # One liked frame becomes the held-out item.
    # Other liked frames form the visual history.
    # --------------------------------------------------------

    liked_frames = user_df[
        user_df["liked"] == 1
    ]["frame_id"].astype(str).tolist()


    if len(liked_frames) < 2:

        continue


    # --------------------------------------------------------
    # Hold out ONE liked frame
    # --------------------------------------------------------

    held_out_frame = liked_frames[0]


    history_frames = [
        frame_id
        for frame_id in liked_frames
        if frame_id != held_out_frame
    ]


    # --------------------------------------------------------
    # Build visual preference from history only
    # --------------------------------------------------------

    history_embeddings = [
        embedding_lookup[frame_id]
        for frame_id in history_frames
        if frame_id in embedding_lookup
    ]


    if len(history_embeddings) == 0:

        continue


    user_visual_vector = np.mean(
        history_embeddings,
        axis=0
    )


    norm = np.linalg.norm(
        user_visual_vector
    )


    if norm > 0:

        user_visual_vector = (
            user_visual_vector / norm
        )


    # --------------------------------------------------------
    # Score every candidate frame
    # --------------------------------------------------------

    candidate_rows = []


    for _, row in user_df.iterrows():

        frame_id = str(
            row["frame_id"]
        )


        # Do not recommend frames already
        # used in the user's history.

        if frame_id in history_frames:

            continue


        frame_vector = embedding_lookup.get(
            frame_id
        )


        if frame_vector is None:

            continue


        visual_score = float(
            np.dot(
                user_visual_vector,
                frame_vector
            )
        )


        # ----------------------------------------------------
        # Metadata score
        # ----------------------------------------------------

        meta_score = metadata_score(
            row
        )


        # ----------------------------------------------------
        # Hybrid score
        # ----------------------------------------------------
        #
        # 70% metadata
        # 30% visual preference
        #
        # This is an evaluation baseline.
        # We will later learn/tune this weight.
        # ----------------------------------------------------

        hybrid_score = (
            0.70 * meta_score
            + 0.30 * visual_score
        )


        candidate_rows.append({

            "frame_id":
                frame_id,

            "liked":
                int(row["liked"]),

            "metadata_score":
                meta_score,

            "visual_score":
                visual_score,

            "hybrid_score":
                hybrid_score
        })


    if len(candidate_rows) == 0:

        continue


    candidates = pd.DataFrame(
        candidate_rows
    )


    # --------------------------------------------------------
    # Rank metadata model
    # --------------------------------------------------------

    metadata_ranked = candidates.sort_values(
        "metadata_score",
        ascending=False
    )


    # --------------------------------------------------------
    # Rank hybrid model
    # --------------------------------------------------------

    hybrid_ranked = candidates.sort_values(
        "hybrid_score",
        ascending=False
    )


    # --------------------------------------------------------
    # Relevance vectors
    # --------------------------------------------------------

    metadata_relevance = (
        metadata_ranked["liked"]
        .tolist()
    )


    hybrid_relevance = (
        hybrid_ranked["liked"]
        .tolist()
    )


    # --------------------------------------------------------
    # Evaluate K = 1, 3, 5
    # --------------------------------------------------------

    for k in [1, 3, 5]:

        metadata_metrics = calculate_metrics(
            metadata_relevance,
            min(k, len(metadata_relevance))
        )


        hybrid_metrics = calculate_metrics(
            hybrid_relevance,
            min(k, len(hybrid_relevance))
        )


        results.append({

            "user_id":
                user_id,

            "held_out_frame":
                held_out_frame,

            "model":
                "Metadata",

            "k":
                k,

            "precision":
                metadata_metrics[0],

            "recall":
                metadata_metrics[1],

            "hit_rate":
                metadata_metrics[2],

            "ndcg":
                metadata_metrics[3]
        })


        results.append({

            "user_id":
                user_id,

            "held_out_frame":
                held_out_frame,

            "model":
                "Hybrid",

            "k":
                k,

            "precision":
                hybrid_metrics[0],

            "recall":
                hybrid_metrics[1],

            "hit_rate":
                hybrid_metrics[2],

            "ndcg":
                hybrid_metrics[3]
        })


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE RAW RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = (
    results_df
    .groupby(
        ["model", "k"]
    )[
        [
            "precision",
            "recall",
            "hit_rate",
            "ndcg"
        ]
    ]
    .mean()
    .reset_index()
)


print("\n==============================================")
print("RANKING RESULTS")
print("==============================================")


print(
    summary.to_string(
        index=False
    )
)


print(
    f"\nRaw results saved to:"
    f"\n{OUTPUT_PATH}"
)


print("\n==============================================")
print("RANKING EVALUATION COMPLETE")
print("==============================================")
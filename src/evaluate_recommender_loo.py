import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data/processed/recommendations/synthetic_hybrid_interactions.csv"
EMBEDDINGS_PATH = PROJECT_ROOT / "data/processed/eyewear/frame_embeddings.npy"
METADATA_PATH = PROJECT_ROOT / "data/processed/eyewear/frame_metadata.csv"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/recommendations/repeated_loo_ranking_evaluation.csv"

df = pd.read_csv(DATA_PATH)
embeddings = np.load(EMBEDDINGS_PATH)
metadata = pd.read_csv(METADATA_PATH)

print()
print("=" * 60)
print("AURA — REPEATED LEAVE-ONE-OUT EVALUATION")
print("=" * 60)

embedding_lookup = {}
for index, row in metadata.iterrows():
    frame_id = str(row["frame_id"])
    vector = embeddings[index]
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    embedding_lookup[frame_id] = vector

def metadata_score(row):
    return (
        0.50 * row["face_compatibility"]
        + 0.25 * row["style_match"]
        + 0.15 * row["color_match"]
        + 0.10 * row["frame_type_match"]
    )

def ndcg_at_k(relevance, k):
    relevance = np.asarray(relevance[:k], dtype=float)
    if len(relevance) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(relevance) + 2))
    dcg = np.sum((2 ** relevance - 1) / discounts)
    ideal = np.sort(relevance)[::-1]
    ideal_dcg = np.sum((2 ** ideal - 1) / discounts)
    if ideal_dcg == 0:
        return 0.0
    return float(dcg / ideal_dcg)

def calculate_metrics(relevance, k):
    relevance = np.asarray(relevance, dtype=int)
    actual_k = min(k, len(relevance))
    if actual_k == 0:
        return 0.0, 0.0, 0, 0.0
    top_k = relevance[:actual_k]
    relevant_total = np.sum(relevance)
    relevant_in_top_k = np.sum(top_k)
    precision = relevant_in_top_k / actual_k
    recall = relevant_in_top_k / relevant_total if relevant_total > 0 else 0.0
    hit_rate = int(relevant_in_top_k > 0)
    ndcg = ndcg_at_k(relevance, actual_k)
    return float(precision), float(recall), hit_rate, float(ndcg)

results = []
users_evaluated = 0
holdouts_evaluated = 0

for user_id in df["user_id"].unique():
    user_df = df[df["user_id"] == user_id].copy()
    liked_frames = (
        user_df[user_df["liked"] == 1]["frame_id"]
        .astype(str)
        .tolist()
    )

    if len(liked_frames) < 2:
        continue

    users_evaluated += 1

    for held_out_frame in liked_frames:
        history_frames = [
            frame_id for frame_id in liked_frames
            if frame_id != held_out_frame
        ]

        history_embeddings = [
            embedding_lookup[frame_id]
            for frame_id in history_frames
            if frame_id in embedding_lookup
        ]

        if not history_embeddings:
            continue

        user_visual_vector = np.mean(history_embeddings, axis=0)
        norm = np.linalg.norm(user_visual_vector)
        if norm > 0:
            user_visual_vector = user_visual_vector / norm

        candidate_rows = []

        for _, row in user_df.iterrows():
            frame_id = str(row["frame_id"])

            if frame_id in history_frames:
                continue

            frame_vector = embedding_lookup.get(frame_id)
            if frame_vector is None:
                continue

            visual_score = float(np.dot(user_visual_vector, frame_vector))
            meta_score = metadata_score(row)
            hybrid_score = 0.70 * meta_score + 0.30 * visual_score

            candidate_rows.append({
                "frame_id": frame_id,
                "liked": int(row["liked"]),
                "metadata_score": meta_score,
                "visual_score": visual_score,
                "hybrid_score": hybrid_score,
            })

        if not candidate_rows:
            continue

        candidates = pd.DataFrame(candidate_rows)

        rankings = {
            "Metadata": candidates.sort_values("metadata_score", ascending=False),
            "Visual": candidates.sort_values("visual_score", ascending=False),
            "Hybrid": candidates.sort_values("hybrid_score", ascending=False),
        }

        for model_name, ranked in rankings.items():
            relevance = ranked["liked"].astype(int).tolist()

            for k in [1, 3, 5]:
                precision, recall, hit_rate, ndcg = calculate_metrics(relevance, k)
                results.append({
                    "user_id": user_id,
                    "held_out_frame": held_out_frame,
                    "model": model_name,
                    "k": k,
                    "precision": precision,
                    "recall": recall,
                    "hit_rate": hit_rate,
                    "ndcg": ndcg,
                })

        holdouts_evaluated += 1

results_df = pd.DataFrame(results)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(OUTPUT_PATH, index=False)

summary = (
    results_df
    .groupby(["model", "k"])[["precision", "recall", "hit_rate", "ndcg"]]
    .mean()
    .reset_index()
)

print()
print("=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)
print(f"Users evaluated: {users_evaluated}")
print(f"Holdouts evaluated: {holdouts_evaluated}")
print()
print(summary.to_string(index=False))
print()
print("=" * 60)
print("RESULTS SAVED")
print("=" * 60)
print(OUTPUT_PATH)
print()
print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

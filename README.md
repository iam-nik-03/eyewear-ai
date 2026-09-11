# AURA — AI Eyewear Intelligence

AURA is an AI-powered personalized eyewear recommendation and virtual styling system that combines computer vision, visual embeddings, machine-learning recommendation, user feedback, natural-language styling explanations, and landmark-based virtual try-on.

## Overview

AURA analyzes a user's uploaded face image, extracts facial characteristics, evaluates eyewear compatibility, ranks frames according to user preferences, learns from feedback, explains recommendations, and provides a virtual try-on preview.

## System Architecture

User Image + Preferences
↓
Face Detection & 478 Facial Landmarks
↓
Facial Geometry Analysis
↓
Face-Shape Prediction
↓
Frame Catalogue + Visual Embeddings
↓
Candidate Generation
↓
Recommendation Ranking
↓
Personalization
↓
AI Stylist Explanation
↓
Virtual Try-On

## Main Components

### Computer Vision

- MediaPipe Face Landmarker
- 478 facial landmarks
- Facial geometry feature extraction
- Eye-distance and face-rotation estimation

### Face-Shape Classification

Facial geometry features include:

- face width
- face height
- aspect ratio
- forehead width
- cheek width
- jaw width
- normalized width ratios

Baseline models were evaluated using a fixed held-out test set.

### Visual Representation

A pretrained ResNet-50 model is used to generate 2048-dimensional visual embeddings for face/frame images.

### Recommendation Engine

The recommendation system evaluates:

- facial compatibility
- visual preference similarity
- style match
- colour match
- frame-type match

The project evaluates:

1. Metadata baseline
2. Visual similarity baseline
3. Weighted hybrid baseline
4. Trained ML hybrid model

### Personalization

Users can like or dislike frames.

Feedback is used to rebuild a visual preference profile from previously liked frames.

### AI Stylist

A deterministic styling explanation layer converts recommendation signals into human-readable explanations describing why a frame was recommended.

### Virtual Try-On

AURA provides a landmark-based 2D virtual try-on system that adapts:

- frame width
- frame position
- rotation
- transparency/compositing

The current implementation is a computer-vision overlay rather than a generative 3D AR system.

## Catalogue

The prototype contains exactly seven eyewear frames.

The catalogue includes metadata such as:

- frame ID
- brand
- frame type
- frame shape
- colour
- material
- style
- gender style
- size

## Evaluation

### Face-Shape Classification

Current locked held-out benchmark:

| Model        | Accuracy | Weighted F1 |
| ------------ | -------: | ----------: |
| Geometry SVM |   45.00% |      43.31% |
| Visual SVM   |   43.50% |      43.46% |
| Hybrid SVM   |   48.80% |      48.68% |

The hybrid representation improved over the geometry-only baseline by 3.8 percentage points on the locked test set.

### Recommendation Ranking

The recommendation system was evaluated using repeated leave-one-out evaluation on a synthetic development interaction dataset.

- 100 synthetic users
- 481 positive holdouts
- K = 1, 3, 5
- Precision@K
- Recall@K
- Hit Rate@K
- NDCG@K

Current results:

| Model           | Precision@1 | NDCG@3 | NDCG@5 |
| --------------- | ----------: | -----: | -----: |
| Metadata        |      82.74% | 92.22% | 92.83% |
| Weighted Hybrid |      81.91% | 91.52% | 92.40% |
| Visual          |      48.86% | 74.01% | 77.69% |
| ML Hybrid       |      72.97% | 87.40% | 88.71% |

The current benchmark shows that metadata-based ranking performs best on this small synthetic development dataset.

These results should **not** be interpreted as real-world customer recommendation accuracy.

## Important Limitations

- The recommendation benchmark uses synthetic interactions.
- The prototype catalogue contains seven frames.
- ResNet-50 embeddings are generic visual representations rather than eyewear-specific embeddings.
- Face-shape labels are subjective and the geometry classifier has limited predictive performance.
- Real-user face prediction should be interpreted probabilistically rather than as an objective face-shape determination.
- Virtual Try-On is a landmark-based 2D overlay, not full 3D augmented reality.
- Larger real-world user interaction datasets are required for production recommendation evaluation.

## Project Structure

```text
eyewear-ai/
├── app/
│   └── app.py
├── config/
├── data/
│   └── catalogue/
├── models/
├── notebooks/
├── src/
│   ├── face analysis
│   ├── embeddings
│   ├── recommendation
│   ├── personalization
│   ├── stylist
│   └── virtual try-on
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

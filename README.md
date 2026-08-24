# SkinSense — Facial Skin Condition Analyzer

An end-to-end machine learning system that classifies visible facial skin concerns (acne, blackheads, dark spots, pores, wrinkles) from a photo, built as an educational tool — not a medical diagnostic.

**Live demo:** https://skinsense-t102.onrender.com/docs

## Problem

Many people lack easy access to basic educational information about common visible skin concerns. SkinSense classifies the concern type from a photo and is explicitly scoped as an educational tool — not a diagnosis, treatment, or severity assessment. Persistent or severe skin issues should always be checked by a dermatologist.

## Tech Stack

- **Model:** MobileNetV2 (transfer learning, ImageNet pretrained) + custom classification head, TensorFlow/Keras
- **API:** FastAPI, served with Uvicorn
- **Testing:** pytest
- **Containerization:** Docker
- **CI/CD:** GitHub Actions (automated testing) + Render (auto-deploy on push)
- **Deployment:** Render (free tier)

## Dataset

~9,700 images across 5 classes (Kaggle: Skin Issues Dataset). After deduplication (see below), 7,241 images remained for training.

## Pipeline

1. **EDA** — verified image integrity, formats, and sizes; visually spot-checked labels
2. **Deduplication** — detected near-duplicate images via perceptual hashing (18-44% of images per class were near-duplicates); removed to prevent data leakage across train/val/test splits
3. **Preprocessing** — stratified 80/10/10 split, resized to 224×224, augmentation (rotation, flip, brightness, zoom) on training data only
4. **Training** — MobileNetV2 with frozen base + trainable classification head; early stopping on validation loss
5. **Evaluation** — 91.9% test accuracy; per-class F1 ranging 0.87 (dark spots) to 0.98 (wrinkles)
6. **Serving** — FastAPI with `/predict`, `/health`, and root endpoints; proper error handling for invalid uploads
7. **Testing** — pytest suite covering health check, root endpoint, and both success/failure prediction paths
8. **Containerization** — Dockerfile for reproducible deployment
9. **CI/CD** — GitHub Actions runs tests on every push; Render auto-deploys on green commits to `main`
10. **Monitoring** — structured logging of prediction class, confidence, and latency per request

## Results

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Acne | 0.91 | 0.89 | 0.90 |
| Blackheads | 0.88 | 0.89 | 0.89 |
| Dark spots | 0.86 | 0.88 | 0.87 |
| Pores | 0.96 | 0.96 | 0.96 |
| Wrinkles | 0.98 | 0.98 | 0.98 |

**Overall test accuracy: 91.9%**

## Known Limitations (Honest Assessment)

- **Domain gap:** the model was trained on curated dataset images. Testing with a real phone-camera photo showed a misclassification (predicted "blackheads" with 97.6% confidence on what was more likely acne) — highlighting the gap between curated-dataset performance and real-world generalization, and the known issue of softmax models being confidently wrong.
- **Dark spots vs. acne/blackheads confusion:** these classes visually overlap in practice (e.g., healed acne marks resemble dark spots), reflected in dark spots having the lowest F1 score.
- **Inference latency on free tier:** ~6-7 seconds per prediction on Render's free CPU tier, versus ~1-3 seconds locally — a resource constraint of the hosting tier, not the model itself.
- **Educational scope only:** not validated for medical use; not a substitute for professional dermatological assessment.

## Running Locally

```bash
git clone https://github.com/prathmeshgawali2006/SkinSense.git
cd SkinSense
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Visit `http://127.0.0.1:8000/docs` for the interactive API.

## Running with Docker

```bash
docker build -t skinsense .
docker run -p 8000:8000 skinsense
```

## Running Tests

```bash
pytest tests/ -v
```
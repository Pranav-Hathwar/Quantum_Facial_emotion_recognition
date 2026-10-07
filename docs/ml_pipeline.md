# ML pipeline

1. **Data** – FER2013 48×48 grayscale, imported to `train/validation/test/<emotion>/`. Class counts and class weights
   are computed from the data (Disgust is rare), and class-weighted cross-entropy is used.
2. **Preprocessing** – face crop → resize (224) → grayscale replicated to 3 channels → ImageNet normalisation.
3. **Backbone** – torchvision ResNet50 (ImageNet weights), `fc` removed → 2048-d. Modes: `feature_extractor` (frozen,
   features cached once as float16 `.npz`, plus a horizontally-flipped training copy) or `fine_tuning` (layer4 or all).
4. **Heads** – A: classical MLP head. A′: classical control. B: hybrid (see quantum_pipeline.md).
5. **Training** – Adam, early stopping on validation macro-F1, seeds recorded, run directory with `best.pt`,
   `history.json`, `metrics.json`, `test_predictions.npz`.
6. **Evaluation** – accuracy, macro/weighted precision/recall/F1, per-class metrics, confusion matrix, training time,
   head and total inference time per face, parameter counts.
7. **Comparison** – `compare.py` aggregates runs (mean ± std over seeds), pairs A and B by seed for an exact McNemar
   test, and writes `comparison.json/md`. Smoke-test runs (random backbone weights) are excluded.

Fair-comparison rules: same cached features, same splits, same optimiser/learning rate/epochs, ≥3 seeds.
FER2013 labels are noisy (human accuracy ≈ 65 %); expect modest absolute numbers from any model.

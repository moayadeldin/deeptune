# Inference

Runs an already-trained DeepTune model (a checkpoint from a completed
`deeptune.py` run) on new data, instead of re-evaluating it against the
holdout split it was trained and tested against. This is what the desktop
app's **Test** page (see [desktop/README.md](../desktop/README.md)) and the
`infer.py` CLI at the repo root are built on.

## Why this isn't just evaluators/*/evaluate*.py again

Every `evaluate_*.py` module under `evaluators/` computes per-sample
predictions internally and then discards them, keeping only aggregate
metrics (`full_metrics.json`) - reasonable for "how good is the model I just
trained," but not for "what does this model predict for this new sample."
They also all hard-require labels: `TextDataset` raises `KeyError` with no
`labels` column, and `evaluators/vision/evaluator.py`'s `TestTrainer.test`
unconditionally moves `labels` onto the device before computing loss. New
data to actually test a model against - especially genuinely new video or
images someone wants classified - may have no known answer at all.

Each module here (`vision.py`, `video.py`, `text.py`, `tabular.py`,
`timeseries.py`) reuses the same model-construction and checkpoint-loading
calls as its `evaluators/`/`embed/` counterpart, but runs its own small
predict loop: skip loss/metrics entirely, keep every per-sample
prediction and confidence, and tolerate missing labels rather than require
them. `data.py` supplies the matching input side: a raw-folder loader that
falls back to an unlabeled read instead of raising when there are no class
subdirectories (`handlers/raw_to_parquet_dataset.py`'s loaders raise in
that case), a single-file loader for the common "just this one clip/image"
case, and a tolerant version of the video CSV/XLSX manifest reader that
doesn't require a `labels` column.

## What's covered per modality

| Modality | Status |
|---|---|
| Images | Every architecture `utils.get_model_cls` resolves (ResNet, DenseNet, Swin, EfficientNet, VGG, ViT, ConvNeXt), PEFT included. **Not SigLIP** - it's built and loaded through an entirely separate code path (`src/vision/siglip.py`, `evaluators/vision/custom_siglip_evaluate.py`) this doesn't wire up yet. |
| Video | Same coverage as images (frame-sampling models share images' code path) plus every native spatio-temporal architecture (`r3d_18`, `mc3_18`, `r2plus1d_18`, `mvit_v2_s`, `swin3d_*`). |
| Text | Multilingual BERT and GPT-2, classification only, matching what DeepTune trains today. |
| Tabular | GANDALF and TabPFN (both from-scratch and finetuning-mode), classification and regression. |
| Timeseries | DeepAR, with caveats below - the one modality where "new data" isn't fully self-contained. |

## DeepAR needs history it can't get from the new data alone

A forecasting model can't predict from nothing: DeepAR needs
`max_encoder_length` rows of history immediately before whatever it
forecasts. `timeseries.py` supplies that from the picked run's own
`data_splits_*/train_split.parquet` and `val_split.parquet` (found by
`desktop/config.py`'s `find_history_paths`) rather than asking the user to
paste that much history into new data by hand - so testing DeepAR is really
"how would this model have predicted these known-but-new observations,"
not open-ended forecasting into a future the model has no covariates for.
If the new data has no target column (or it's all missing), a `0.0`
placeholder is used so `TimeSeriesDataSet` can still build the decoder
window - DeepAR's decoder-side forward pass doesn't consume that value,
only compares against it afterwards for metrics, which is why `has_labels`
is tracked separately and predictions in that case are reported with no
ground truth rather than a fabricated one. This is the least
battle-tested modality here; treat its predictions with commensurately
more scrutiny than the others.

## Files

- `data.py` - loads new data into the same in-memory DataFrame shape
  training uses (`images`/`videos` byte columns, or a plain tabular frame),
  tolerating missing labels.
- `common.py` - `summarize_predictions`, shared aggregate accuracy/MAE-RMSE
  logic every modality's predictions feed through.
- `vision.py`, `video.py`, `text.py`, `tabular.py`, `timeseries.py` - one
  `predict_*` function per modality/model family, each returning a
  DataFrame: `sample_id`, `predicted_class`/`predicted_label`/
  `predicted_value`, `confidence` (classification only), and
  `true_label`/`correct` or `true_value`/`abs_error` when the input had
  labels.
- `../infer.py` - the CLI entry point tying a modality's `predict_*`
  function to `--checkpoint`/`--data`/`--out`, writing `predictions.csv`
  and, only when there was ground truth, `inference_metrics.json`.

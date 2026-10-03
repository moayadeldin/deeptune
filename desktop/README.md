# DeepTune desktop

Run from the repository root with the project environment:

```powershell
.venv\Scripts\python.exe desktop\app.py
```

The interface uses Dear PyGui; the run itself still goes through deeptune.py's
own one-call CLI, launched as a subprocess (see `desktop/runner.py`) - the GUI
process only imports pandas and dearpygui, never torch/transformers, so it
opens quickly and never blocks on the ML stack just to show you a form.

## Pages

- **Overview** - what the app does and where to start.
- **Dataset** - pick a modality (images / text / tabular / timeseries /
  video), point at a raw data folder or an existing parquet file, and preview
  it. Raw images/video folders are previewed with a per-class file count using
  the exact same directory convention `handlers/raw_to_parquet_dataset.py`
  expects (one subdirectory per class, optionally grouped under split
  subdirectories). Video also supports selecting individual MP4 files and
  building a labeled clip list directly in the app. Other modalities expect
  a single CSV/XLSX file, or an existing parquet.
- **Model & Training** - model version (the list updates for the chosen
  modality), transfer-learning options (added layers, embedding size, freeze
  backbone, PEFT/LoRA), training hyperparameters (batch size, epochs,
  learning rate), and whatever else that modality/model needs: frame count
  and pooling for video, continuous/categorical columns for GANDALF,
  fine-tuning mode for TabPFN, the time index column for DeepAR.
- **Run** - choose an output directory, review the equivalent CLI command,
  launch it, and follow the live (tqdm-aware) log. Stop cancels the run and
  its worker processes. Save or load reusable JSON settings; the last session
  is restored when you reopen the app. Full logs and the configuration for
  each launch are saved automatically under `desktop-sessions/` in the output folder.
- **Results** - completed and partial runs under the chosen output directory, with
  its holdout metrics table, training/validation loss curve, checkpoint path,
  and embeddings file. Export all available metrics to CSV and open the selected
  experiment folder. Incomplete files are reported without breaking the page.
- **Test** - load a completed run's checkpoint and run it on new data instead of
  the holdout split it was trained against: point at a single new image or video
  file, a folder of them (one subfolder per class to compare against known
  labels, or a flat folder to just predict), or a new CSV/XLSX/parquet file for
  text/tabular/timeseries. Get each sample's predicted class or value,
  confidence, and - when the new data has its own labels - whether the
  prediction was correct and an overall
  accuracy (or MAE/RMSE for regression). Model version, class count, and every
  other model-build setting come from the picked run's own recorded settings,
  not re-entered by hand. Runs through `infer.py` the same way the Run page
  launches `deeptune.py`: a subprocess with a live log, cancellable, writing
  `predictions.csv` (and `inference_metrics.json`, when there was ground truth
  to compare against) under that run's own folder rather than the training
  output layout. SigLIP images and DeepAR timeseries have narrower support
  today - see `inference/README.md` for exactly what's covered per modality
  and why.

Previews load on a worker thread and read at most 50 table rows. Binary media
payloads are shown as byte counts. Model controls reflect supported combinations;
for example, TabPFN exposes regression, GPT-2 disables PEFT, and MViT uses 16 frames.
Quick-check and standard-training presets set epochs, batch size, and backbone freezing.

The default split is 70% training, 10% validation, and 20% holdout. Classification
splits are stratified; grouped splits keep each group together, with approximate
ratios. Forecasting splits follow time order. Class folders are classification
inputs; use parquet with numeric targets for image/video regression, or a
video CSV/XLSX list with numeric labels for video regression.

Native video models use the preprocessing supplied with their Kinetics weights.
Frame models use the image backbone preprocessing. Model weights may download on
the first run. TabPFN weights may require Hugging Face access; DeepTune uses your
existing authentication and does not replace it. Desktop runs report missing
authentication in the log instead of opening an interactive login prompt.

## Import MP4 videos

1. On **Dataset**, choose **Video** and keep **Raw data** enabled.
2. Click **Browse file** or **Select MP4 / video files**, select one or more
   clips (Ctrl/Shift for multiple selections), then confirm the file picker.
   The files appear in the list immediately, without requiring a label.
   DeepTune opens the video preview and starts muted, looping playback with
   play/pause, restart, and a seek timeline. It shows dimensions, frame rate, and duration;
   any import or decoding error appears beside the import controls.
3. Type a class label beside each clip, or enter one in the shared field and
   click **Apply label to selected**. A class label describes what the video
   shows, such as walking or running. It is needed for training, not previewing.
4. Click **Use clip dataset**, choose training settings, and launch from **Run**.

Click **Preview** beside any imported clip to play it without changing the
training dataset path. The preview pauses when you leave the Dataset page.
For audio or full-screen playback, use **Open in video player** to launch the
file in your system's video player. Inline playback uses OpenCV's bundled
FFmpeg decoder; a separate FFmpeg installation is not required for the verified
Windows environment. Decoding runs on a worker, and only the latest display
frame is retained, so the preview does not preload an entire video into memory.

The clip list is saved as a CSV in your local DeepTune settings directory.
Keep the original videos available at their selected paths. Loading that CSV
restores the list for editing. For a useful split, start with at least ten
clips per class; training and validation need examples from each class.

You can also supply a CSV/XLSX file containing `videos` (local file paths) and
`labels` columns. Relative paths are resolved against the list's directory.
Duplicate files and missing clips are rejected. MP4, AVI, MOV, MKV, WebM, M4V,
WMV, MPEG, and MPG containers are accepted; decoding depends on the codec
inside the file. Corrupt or unsupported clips produce a readable error.

## Development checks

```powershell
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe desktop\app.py --self-test --data <raw_data_folder> --modality video --outdir desktop-smoke-results
```

This builds the UI, exercises every nav page and modality switch through a
real (briefly shown) viewport, then launches one real, tiny training run
(1 epoch, a frozen backbone) through the exact same runner the GUI's Run
button uses, and confirms it produced a normal results folder. It needs a
small real image/video dataset on disk (at least 10 examples per class is a useful
starting point for the default split). It checks for metrics, a checkpoint, and
embeddings and stops after ten minutes if the run has not finished.

For a repeatable video plumbing check using generated moving-shape clips:

```powershell
.venv\Scripts\python.exe scripts\smoke_video.py --model resnet18
.venv\Scripts\python.exe scripts\smoke_video.py --model resnet18 --manifest
.venv\Scripts\python.exe scripts\smoke_video.py --model r3d_18
.venv\Scripts\python.exe scripts\smoke_video.py --model resnet18 --peft --pooling attention
```

These smoke tests validate execution and artifact shapes, not predictive quality.

## Packaging

Python 3.12 is the verified environment. Install with `python -m pip install .`;
the package provides `deeptune` and `deeptune-desktop` commands. Build a wheel with
`python -m pip wheel . --no-deps --wheel-dir dist`.

### Standalone Windows installer

For end users who shouldn't need to install Python themselves, `packaging/`
builds a standalone `DeepTune-Setup-<version>.exe`: a normal Windows
installer that unpacks a self-contained copy of the app (PyInstaller-frozen,
with its own bundled Python and every dependency, including torch/
transformers/tabpfn) and adds Start Menu / Desktop shortcuts. See
[packaging/README.md](../packaging/README.md) for how to build it and what
it needs on the machine that runs it.

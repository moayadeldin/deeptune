# DeepTune verification — 9 September 2026

Verified locally on Windows with Python 3.12.10 and the dependencies in
`requirements.txt`.

| Check | Result |
| --- | --- |
| Automated suite | 117 passed, 3 warnings; no failures |
| Dependency consistency | `pip check` passed |
| Wheel build and isolated package installation | Passed; CLI and desktop help commands work outside the source root |
| Desktop rendering | All five pages inspected; MP4 file picker, clip list, and decoded frame preview inspected |
| MP4 selection and labeled dataset | Actual uppercase `.MP4` files decoded in UI tests; generated CSV preserves paths and labels and restores the clip list |
| Video pipeline: ResNet18, mean pooling | Training, evaluation, checkpoint, and embeddings passed |
| Video pipeline: ResNet18, attention pooling + PEFT | Training, evaluation, checkpoint, and embeddings passed |
| Video pipeline: native R3D-18, with and without PEFT | Training, evaluation, checkpoint, and embeddings passed |
| Video pipeline: individual MP4 file list | Training, evaluation, checkpoint, and embeddings passed |
| GANDALF | Actual training/evaluation run passed; embedding extraction retains every holdout row across batches |
| DeepAR | Actual training/evaluation/embedding run passed with time-ordered splits and saved preprocessing |
| TabPFN | End-to-end verification blocked by external model access/network availability |

The video smoke runs use 40 generated MP4 clips, two classes, and one epoch with
a frozen pretrained backbone. Each produced an 8-row holdout embedding table
with 16 embedding features and a label. They verify execution and output
consistency, not accuracy on a real task. Other architectures have automated
coverage where supplied by the test suite; every pretrained architecture has
not been trained end to end.

The TabPFN attempt reached its model download/license check. The last attempt
reported a DNS lookup failure followed by the library's gated-model access
error. Desktop subprocesses disable interactive browser login; existing
authentication remains available to them. Its model weights and network access
are needed before this path can be verified fully.

Run the checks again from the repository root:

```powershell
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe scripts\smoke_video.py --model resnet18 --manifest
.venv\Scripts\python.exe scripts\smoke_video.py --model r3d_18
.venv\Scripts\python.exe scripts\smoke_video.py --model resnet18 --peft --pooling attention
.venv\Scripts\python.exe -m pip wheel . --no-deps --wheel-dir dist
```

Local evidence is under `.artifacts/`: `test-results.xml`,
`tests-console.log`, `ui/`, and experiment outputs under `video-smoke/`,
`tabular-smoke/`, and `timeseries-smoke/`. These generated artifacts are ignored
by Git. See [the desktop guide](../desktop/README.md) for MP4 import steps.

## Follow-up: selected videos did not appear in the list

Selecting a video now imports it into the visible list immediately, with an
editable class label. A label is only required when preparing the training
dataset. The previous Add action required a label before showing the file,
and reported missing labels only in the sidebar. Messages for missing labels,
missing files, and decoding errors now appear beside the import controls.

The follow-up passed 38 focused UI, manifest, and workflow checks. The tests
dispatch the file dialog and labeling/save button callbacks registered with
Dear PyGui, decode actual MP4 files, and verify the saved dataset. Additional
coverage checks that background previews do not overwrite label edits. The
revised panel was rendered and inspected with both empty and completed labels.
Actual mouse/keyboard automation could not be checked because the Windows
computer-use helper's native pipe was unavailable.

## Follow-up: video playback

The preview now uses a dynamic texture updated from a background decoder,
replacing the first-frame-only display. It starts muted playback, with
play/pause, restart, looping, and seeking. The clip list's Preview button plays
an individual video without replacing the training dataset path. Navigating
away pauses playback; changing videos or closing the app releases the decoder.
Open in video player launches the system player when audio is wanted.

OpenCV's build information confirms that this Windows environment already
includes FFmpeg. No separate decoder installation was added. The worker keeps
only its newest preview frame, rather than buffering the full video.

Fifteen player/UI tests passed after the final layout change. These cover
changing pixels over time, pause/resume, seeking while paused, end-of-file,
replay, looping, bounded frame delivery, corrupt files, decoder cleanup, and
preserving the training source. Related manifest/workflow checks also passed.
An actual rendered viewport showed frames advancing, then stayed on frame 51
after pausing and seeking to 2.5 seconds in a generated 20-fps MP4. Screenshots
are in `.artifacts/ui/video-playing-early.png`, `video-playing-later.png`, and
`video-paused-seek.png`. These checks use generated clips, not the user's video.

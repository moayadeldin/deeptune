# Packaging DeepTune as a Windows installer

Turns the desktop app into `DeepTune-Setup-<version>.exe`: a normal Windows
installer for people who don't have (and shouldn't need) Python, a virtual
environment, or `pip install -r requirements.txt`. It bundles a full copy of
the project's dependencies -- torch, transformers, tabpfn,
pytorch-lightning, pytorch-tabular, pytorch-forecasting, opencv, and the
rest of `requirements.txt` -- so it is large and takes a while to build.
There's no shortcut around that; it's the cost of "double-click to run,
nothing else to install."

## Building it

Requires [Inno Setup 6](https://jrsoftware.org/isinfo.php) (`ISCC.exe`) on
the machine doing the build, and a `.venv` at the repo root with
`requirements.txt` already installed (`pyinstaller` and
`pyinstaller-hooks-contrib` are installed into it automatically if missing).
From the repo root, in PowerShell:

```powershell
packaging\build_installer.ps1
```

That runs two steps, each also usable on its own:

1. `packaging\build_exe.ps1` -- PyInstaller freezes `desktop/app.py` per
   `packaging\deeptune.spec` into `packaging\dist\DeepTune\DeepTune.exe`
   plus a sibling `_internal\` folder holding every dependency (a "onedir"
   build, not a single onefile exe -- see the comment at the top of
   `deeptune.spec` for why). Expect this step to take several minutes and
   produce a multi-gigabyte folder; budget free disk space accordingly (a
   clean build plus its intermediate `packaging\build\` work directory has
   used on the order of 5-8 GB during development).
2. `packaging\installer.iss` -- Inno Setup compresses that folder into
   `packaging\installer-dist\DeepTune-Setup-<version>.exe`, the one file to
   actually share. The version number comes from the repo's `VERSION` file
   unless you pass `-Version`.

Re-run `build_installer.ps1 -SkipBuild` to re-package an existing
`packaging\dist\DeepTune` build (e.g. after only editing `installer.iss`)
without waiting through the PyInstaller step again. Pass `-Clean` to wipe
previous PyInstaller output first if a build looks stale or broken.

`packaging\dist\`, `packaging\build\`, and `packaging\installer-dist\` are
all build output (matched by the repo's existing `dist/`/`build/`
`.gitignore` entries) -- never commit them; only `deeptune.spec`,
`installer.iss`, `build_exe.ps1`, `build_installer.ps1`, and `assets/` are
checked in.

## Verifying a build

A build reporting success from PyInstaller is not proof it actually works --
two real bugs (torchvision's native extension silently missing, and
`pytorch_lightning`/`lightning_fabric` missing a required data file) only
surfaced by actually running the frozen exe's worker mode, not from the
build log. Before trusting a build:

1. **Fast import check** (seconds): run the frozen exe's worker mode with
   deliberately incomplete arguments, so it exercises the full
   torch/transformers/tabpfn/pytorch-lightning/... import chain and then
   fails on argparse validation rather than training anything:
   ```powershell
   packaging\dist\DeepTune\DeepTune.exe --deeptune-worker --modality images --out packaging\smoke-out
   ```
   A clean `usage: ...` message and `error: the following arguments are
   required: --df` means every import along that path succeeded. A
   `Traceback`/`ModuleNotFoundError`/`RuntimeError` instead means something
   is still missing from `deeptune.spec`.
2. **Real end-to-end run** (a minute or two): actually train a tiny model
   and, for the Test page, run inference against it, the same way
   `desktop/app.py --self-test` does for the source checkout. GANDALF is
   the fastest modality to use for this (no pretrained-weight download, no
   GPU needed) -- a few dozen synthetic rows is enough:
   ```powershell
   $env:PYTHONIOENCODING = "utf-8"
   packaging\dist\DeepTune\DeepTune.exe --deeptune-worker --modality tabular --model_version gandalf --raw-data --df <small.csv> --out <out> --target labels --type classification --continuous_cols <cols> --categorical_cols <cols> --num_epochs 1 --batch_size 8 --fixed-seed
   packaging\dist\DeepTune\DeepTune.exe --infer-worker --modality tabular --checkpoint <out>\deeptune-*\trainval_output_GANDALF_*\GANDALF_model --data <new.csv> --out <infer-out> --mode cls --target labels --model_version gandalf
   ```
   Confirm training's own holdout metrics print, and that
   `<infer-out>\predictions.csv` has sensible per-row predictions. Setting
   `PYTHONIOENCODING` is only needed for a console you're watching directly;
   the app itself doesn't rely on it (see the stdout-reconfigure note in
   `desktop/app.py`, right where `--deeptune-worker`/`--infer-worker` are
   handled).

Both of the bugs above were only caught by step 2's real run completing
end to end -- step 1 alone would have missed the
`pytorch_lightning`/`lightning_fabric` issue, since importing those
succeeded fine; only building the actual trainer failed on their missing
`version.info` file.

## How the frozen build runs training and inference without a separate Python

Normally the desktop app stays light (only `pandas` + `dearpygui` loaded)
and launches `deeptune.py` (training) or `infer.py` (the Test page's
prediction-on-new-data run -- see [inference/README.md](../inference/README.md))
as a subprocess of the real `python.exe` for the actual work -- see
`desktop/runner.py`. A frozen build has no separate `python.exe` to hand
either script to as a script path: `sys.executable` there is `DeepTune.exe`
itself. So instead:

- `desktop/app.py` checks, before importing DearPyGui, whether it was
  launched as `DeepTune.exe --deeptune-worker <deeptune.py's own argv>` or
  `DeepTune.exe --infer-worker <infer.py's own argv>`; if so it calls
  `deeptune.main()`/`infer.main()` directly in that process instead of
  building the GUI.
- `desktop/runner.py` detects `sys.frozen` (set by PyInstaller) and, only in
  that case, launches `[sys.executable, "--deeptune-worker"|"--infer-worker", *args]`
  instead of `[sys.executable, "-u", "<path to deeptune.py|infer.py>", *args]`
  (see its `_worker_command` helper, shared by both `start_run` and
  `start_infer_run`).
- `multiprocessing.freeze_support()` is called at the top of `app.py`'s
  `if __name__ == "__main__":` block, as Python's own docs require for any
  frozen Windows executable that spawns subprocesses via `multiprocessing`
  (e.g. a PyTorch `DataLoader` with `num_workers > 0`).

`infer.py` and the `inference/` package it dispatches to introduce no new
third-party dependencies beyond what `deeptune.spec` already collects for
training/evaluation, so no spec changes were needed for the Test page --
PyInstaller's static analysis picks up `infer.py`'s import graph the same
way it already does `deeptune.py`'s, by following the `import infer`
inside `app.py`'s `--infer-worker` branch. Re-run the smoke test in
"Verifying a build" below after packaging changes to confirm that's still
true after any future changes to `inference/`.

None of this changes behavior when running from source (`python
desktop/app.py`); `sys.frozen` is only set inside a PyInstaller build.

## What the installer needs from the machine it installs on

- **Windows 10 or 11, 64-bit.** Nothing else is built (no macOS/Linux
  installer here); people on other platforms should run from source instead
  (see the main [README](../README.md)) or use their own PyInstaller build.
- **No Python, and no admin rights.** The installer writes to the current
  user's `%LOCALAPPDATA%\Programs\DeepTune`, not Program Files, so a
  standard (non-administrator) Windows account can install and run it.
- **Free disk space:** several GB for the installed app itself (the ML
  dependency stack), plus room to grow for the datasets, checkpoints, and
  results the app writes to whatever output folder the user picks, plus the
  Hugging Face model cache under `%USERPROFILE%\.cache\huggingface` for any
  pretrained weights downloaded on first use of a given model.
- **RAM:** 8 GB is a practical floor for training the smaller models on
  small datasets; 16 GB+ is safer for larger models, batch sizes, or video.
  This is a CPU-only build (see below), and CPU training is far more
  memory- and time-hungry than GPU training for the same job.
- **No GPU required -- and none used even if present.** `requirements.txt`
  pins the CPU build of torch/torchvision, so DeepTune here always trains on
  CPU, which is slow for anything beyond small models/datasets. There is
  currently no CUDA build of this installer; someone wanting GPU-accelerated
  training needs to run DeepTune from source in their own environment with
  a CUDA-enabled torch install instead.
- **Internet access, at least the first time a given model is used.**
  Several models (anything transformers-/tabpfn-based) download pretrained
  weights from Hugging Face on first use and cache them locally; after
  that, the same model works offline. A network that blocks
  `huggingface.co` will make those models fail on first use with a clear
  error in the run log rather than an interactive login prompt (see the
  main desktop README's note on TabPFN authentication).
- **A Windows SmartScreen prompt on first run.** The installer and
  `DeepTune.exe` are not code-signed (that requires a paid certificate this
  project doesn't have), so Windows will show an "Unknown publisher"
  warning the first time someone runs either one. Click **More info -> Run
  anyway** to proceed; this is expected and not a sign of a bad download,
  but it's worth telling people in advance so they don't assume the file is
  corrupted or malicious.

# PyInstaller spec for the DeepTune desktop app.
#
# Builds a "onedir" bundle (an EXE plus its dependencies in a sibling
# _internal/ folder) rather than a single onefile EXE: this project's
# dependency closet (torch, transformers, tabpfn, pytorch-lightning,
# pytorch-tabular, pytorch-forecasting, opencv, ...) is large, and onefile
# would re-extract all of it into a temp directory on every launch. The
# installer (packaging/installer.iss) packages this whole folder, so users
# never see the difference -- they only ever run the installer's one EXE.
#
# Build with packaging/build_exe.ps1, or directly:
#   .venv\Scripts\python.exe -m PyInstaller packaging\deeptune.spec ^
#       --distpath packaging\dist --workpath packaging\build --noconfirm
#
# Entry point: desktop/app.py. Its very first branch (before the DearPyGui
# import) checks for a "--deeptune-worker" sentinel in argv and, if found,
# calls deeptune.main() directly instead of building the GUI -- see the
# comment there and in desktop/runner.py for why: a frozen build has no
# separate python.exe for runner.py to hand deeptune.py to as a script
# path, so it re-invokes this same EXE in that mode instead.
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata

ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 (SPECPATH is injected by PyInstaller; packaging/ sits directly under the repo root)
APP_NAME = "DeepTune"
ICON = str(ROOT / "packaging" / "assets" / "deeptune.ico")

# ---------------------------------------------------------------------------
# Third-party packages PyInstaller's static import analysis can't fully see
# on its own: heavy use of plugin discovery, importlib, or non-.py data
# files. pyinstaller-hooks-contrib (installed alongside pyinstaller) already
# ships hooks for torch/torchvision/transformers/sklearn/cv2/lightning/
# pyarrow/openpyxl, discovered automatically -- these are the rest.
# ---------------------------------------------------------------------------
COLLECT_ALL_PACKAGES = [
    "tabpfn",
    "tabpfn_extensions",
    "tabpfn_common_utils",
    "peft",
    "accelerate",
    "pytorch_tabular",
    "pytorch_forecasting",
    "huggingface_hub",
    "tokenizers",
    "safetensors",
    "tensorboard",
    "dearpygui",
    "omegaconf",
    "hyperopt",
    "einops",
    "joblib",
    "tabulate",
    "torchmetrics",
    # "lightning" (hooks-contrib's hook-lightning.py covers this one) wraps
    # two other, separately-distributed top-level packages this codebase
    # imports directly (helpers.py: pytorch_lightning; train_deepar.py:
    # lightning.pytorch) or transitively (pytorch_lightning's own __init__
    # imports lightning_fabric) -- each ships its own version.info data
    # file at its package root that only collect_all() per-package finds;
    # without it, import fails with FileNotFoundError on version.info,
    # same root cause as the torchvision fix above but for a plain data
    # file rather than a native extension.
    "pytorch_lightning",
    "lightning_fabric",
]

# Distribution (PyPI) names for packages whose code looks up its own
# installed version via importlib.metadata at runtime -- collect_all()
# above bundles code and data files but not the .dist-info metadata that
# powers those lookups, so those packages raise PackageNotFoundError
# without this even though every module imports fine.
METADATA_PACKAGES = [
    "torch", "torchvision", "transformers", "tokenizers", "huggingface_hub",
    "safetensors", "accelerate", "peft", "tabpfn", "tabpfn-extensions",
    "tabpfn-common-utils", "lightning", "pytorch-lightning", "pytorch_tabular",
    "pytorch-forecasting", "torchmetrics", "lightning-utilities", "tqdm",
    "regex", "requests", "packaging", "filelock", "numpy", "PyYAML",
    "tensorboard", "tensorboard-data-server", "scikit-learn", "scipy",
    "pandas", "pyarrow", "joblib", "omegaconf", "hyperopt", "einops",
    "opencv-python", "pillow", "dearpygui", "Markdown", "protobuf", "grpcio",
    "Werkzeug",
]

datas = []
binaries = []
hiddenimports = [
    # trainers/nlp/train_multilingualbert.py resolves these two by name via
    # importlib.import_module() rather than a static import, so PyInstaller
    # can't discover them by scanning imports the way it does everywhere
    # else in this codebase.
    "src.nlp.multilingual_bert",
    "src.nlp.multilingual_bert_peft",
]

for _pkg in COLLECT_ALL_PACKAGES:
    _d, _b, _h = collect_all(_pkg)
    datas += _d
    binaries += _b
    hiddenimports += _h

# pyinstaller-hooks-contrib's torchvision hook (as of 2026.7) still looks for
# this package's native extension under its old names ("_C"/"image") and
# warns "Hidden import ... not found" without bundling anything -- this
# torchvision release ships them as "_C_stable"/"image_stable" (its
# stable-ABI extension, loaded by torch.ops.load_library() from a file path
# in torchvision/extension.py, not a normal Python import, which is also why
# a hiddenimports entry wouldn't fix this even with the right name). Without
# these files, torchvision imports "successfully" but every operator it
# registers is missing, so anything using torchvision.models (i.e. every
# vision/video backbone in this project) fails at import time with
# "RuntimeError: operator torchvision::nms does not exist". Copy the
# extension .pyd files and their .dll dependencies (libjpeg/libpng/libwebp/
# zlib) straight from the venv's torchvision install to sidestep the hook
# rather than wait on it to catch up.
import torchvision as _torchvision_probe  # noqa: E402 (spec files aren't regular modules)

_TORCHVISION_DIR = Path(_torchvision_probe.__file__).resolve().parent
for _pattern in ("*.pyd", "*.dll"):
    for _f in _TORCHVISION_DIR.glob(_pattern):
        binaries.append((str(_f), "torchvision"))

_missing_metadata = []
for _pkg in METADATA_PACKAGES:
    try:
        datas += copy_metadata(_pkg)
    except Exception:
        _missing_metadata.append(_pkg)
if _missing_metadata:
    print(f"[deeptune.spec] no installed metadata for: {', '.join(_missing_metadata)} "
          f"(skipped -- fine if that package doesn't check its own version at runtime)")

block_cipher = None

a = Analysis(
    [str(ROOT / "desktop" / "app.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=ICON,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)

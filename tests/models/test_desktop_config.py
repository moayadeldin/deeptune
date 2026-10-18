import json
from pathlib import Path

import pandas as pd
import pytest

from desktop.config import (
    MODEL_VERSIONS_BY_MODALITY,
    InferenceConfig,
    RunConfig,
    RunResults,
    build_argv,
    build_infer_argv,
    find_history_paths,
    find_latest_run,
    infer_config_from_run,
    list_runs,
    read_inference_results,
    read_run_results,
    validate_config,
    validate_infer_config,
)


def test_build_argv_images_config():
    cfg = RunConfig(
        modality="images", df=Path("data/raw"), raw_data=True, out=Path("out"),
        model_version="resnet50", num_classes=3, mode="cls", added_layers=2,
        embed_size=512, freeze_backbone=True, use_peft=False, fixed_seed=True,
        batch_size=32, num_epochs=5, learning_rate=1e-3,
    )
    argv = build_argv(cfg)
    assert "--modality" in argv and argv[argv.index("--modality") + 1] == "images"
    assert argv[argv.index("--model_version") + 1] == "resnet50"
    assert "--raw-data" in argv
    assert "--freeze-backbone" in argv
    assert "--use-peft" not in argv
    assert argv[argv.index("--num_epochs") + 1] == "5"


def test_build_argv_video_includes_frame_options():
    cfg = RunConfig(modality="video", df=Path("clips"), out=Path("out"),
                     model_version="r3d_18", num_frames=16, pooling="attention")
    argv = build_argv(cfg)
    assert argv[argv.index("--num_frames") + 1] == "16"
    assert argv[argv.index("--pooling") + 1] == "attention"


def test_build_argv_gandalf_includes_columns_and_type():
    cfg = RunConfig(modality="tabular", df=Path("data.parquet"), raw_data=False, out=Path("out"),
                     model_version="gandalf", gandalf_type="regression",
                     continuous_cols=["age", "bmi"], categorical_cols=["sex"])
    argv = build_argv(cfg)
    assert argv[argv.index("--type") + 1] == "regression"
    cont_idx = argv.index("--continuous_cols")
    assert argv[cont_idx + 1 : cont_idx + 3] == ["age", "bmi"]
    cat_idx = argv.index("--categorical_cols")
    assert argv[cat_idx + 1] == "sex"


def test_build_argv_tabpfn_finetuning_flag():
    cfg = RunConfig(modality="tabular", df=Path("data.parquet"), raw_data=False, out=Path("out"),
                     model_version="tabpfn", finetuning_mode=True)
    argv = build_argv(cfg)
    assert "--finetuning-mode" in argv
    assert "--type" not in argv  # gandalf-only


def test_build_argv_timeseries_time_idx_column():
    cfg = RunConfig(modality="timeseries", df=Path("data.parquet"), raw_data=False, out=Path("out"),
                     model_version="deepAR", time_idx_column="time_idx")
    argv = build_argv(cfg)
    assert argv[argv.index("--time_idx_column") + 1] == "time_idx"


def test_validate_config_flags_missing_dataset_and_outdir():
    cfg = RunConfig(df=None, out=None)
    issues = validate_config(cfg)
    assert any("dataset" in issue.lower() for issue in issues)
    assert any("output directory" in issue.lower() for issue in issues)


def test_validate_config_flags_missing_dataset_path(tmp_path):
    cfg = RunConfig(df=tmp_path / "does_not_exist", out=tmp_path)
    issues = validate_config(cfg)
    assert any("does not exist" in issue for issue in issues)


def test_validate_config_passes_for_well_formed_config(tmp_path):
    dataset = tmp_path / "data"
    dataset.mkdir()
    cfg = RunConfig(modality="images", df=dataset, out=tmp_path / "out", model_version="resnet18",
                     num_classes=2, mode="cls", added_layers=2)
    assert validate_config(cfg) == []


def test_video_model_versions_exclude_siglip():
    # siglip has no entry in utils.MODEL_CLS_MAP, so it cannot back a
    # frame-sampling video model; only the native video architectures and
    # the other 2D backbones should be offered.
    assert "siglip" not in MODEL_VERSIONS_BY_MODALITY["video"]
    assert "r3d_18" in MODEL_VERSIONS_BY_MODALITY["video"]
    assert "resnet50" in MODEL_VERSIONS_BY_MODALITY["video"]


def _make_run_dir(out_dir: Path, name: str) -> Path:
    run_dir = out_dir / name
    (run_dir / "trainval_output_resnet18_20260101_0000").mkdir(parents=True)
    (run_dir / "eval_output_resnet18_20260101_0000").mkdir(parents=True)
    (run_dir / "embed_output_resnet18_20260101_0000").mkdir(parents=True)
    (run_dir / "data_splits_20260101_0000").mkdir(parents=True)

    metrics = {"loss": 0.42, "accuracy": 0.9, "0": {"precision": 1.0, "recall": 0.8, "f1-score": 0.89, "support": 5}}
    (run_dir / "eval_output_resnet18_20260101_0000" / "full_metrics.json").write_text(json.dumps(metrics))

    training_log = pd.DataFrame({
        "epoch": [1, 2], "epoch_loss": [0.9, 0.5], "epoch_accuracy": [50.0, 80.0],
        "val_loss": [0.95, 0.6], "val_accuracy": [45.0, 75.0],
    })
    training_log.to_csv(run_dir / "trainval_output_resnet18_20260101_0000" / "training_log.csv", index=False)

    (run_dir / "trainval_output_resnet18_20260101_0000" / "model_weights.pth").write_bytes(b"fake-weights")

    (run_dir / "data_splits_20260101_0000" / "label_mapping.json").write_text(json.dumps({"cat": 0, "dog": 1}))

    embed_df = pd.DataFrame({0: [0.1, 0.2], 1: [0.3, 0.4], "label": [0, 1]})
    embed_df.to_parquet(run_dir / "embed_output_resnet18_20260101_0000" / "resnet18_cls_embeddings.parquet")

    return run_dir


def test_list_runs_and_find_latest_run(tmp_path):
    _make_run_dir(tmp_path, "deeptune-20260101-exp1")
    latest = _make_run_dir(tmp_path, "deeptune-20260101-exp2")

    runs = list_runs(tmp_path)
    assert len(runs) == 2
    assert find_latest_run(tmp_path) in runs

    # a directory that doesn't match DeepTune's own run-folder naming
    # convention should be ignored
    (tmp_path / "not_a_run").mkdir()
    assert len(list_runs(tmp_path)) == 2
    assert list_runs(tmp_path / "missing") == []


def test_read_run_results_parses_everything(tmp_path):
    run_dir = _make_run_dir(tmp_path, "deeptune-20260101-exp1")
    results = read_run_results(run_dir)

    assert results.metrics["accuracy"] == 0.9
    assert list(results.training_log["epoch_loss"]) == [0.9, 0.5]
    assert results.label_mapping == {"cat": 0, "dog": 1}
    assert results.checkpoint_path is not None and results.checkpoint_path.exists()
    assert results.embeddings_path is not None and results.embeddings_path.exists()
    assert results.embeddings_shape == (2, 3)


def test_read_run_results_missing_pieces_are_none(tmp_path):
    run_dir = tmp_path / "deeptune-20260101-exp1"
    run_dir.mkdir()
    results = read_run_results(run_dir)
    assert results.metrics is None
    assert results.training_log is None
    assert results.checkpoint_path is None
    assert results.embeddings_path is None


# ---------------------------------------------------------------------------
# Test/Inference (InferenceConfig, build_infer_argv, validate_infer_config,
# infer_config_from_run, find_history_paths, read_inference_results) -- see
# inference/ and infer.py for the code these feed into.
# ---------------------------------------------------------------------------

def test_infer_config_from_run_fills_defaults_for_null_fields():
    # GANDALF/tabular runs record num_classes/added_layers/embed_size as
    # JSON null (they don't apply) -- infer_config_from_run must fall back
    # to sane defaults rather than propagate None into an int field.
    results = RunResults(
        run_dir=Path("run"), checkpoint_path=Path("run/trainval/GANDALF_model"),
        cli_arguments={"modality": "tabular", "model_version": "gandalf", "mode": "cls",
                       "num_classes": None, "added_layers": None, "embed_size": None,
                       "target": "labels"},
    )
    cfg = infer_config_from_run(results, data=Path("new.csv"), out=Path("out"))
    assert cfg.modality == "tabular"
    assert cfg.num_classes == 2
    assert cfg.added_layers == 2
    assert cfg.embed_size == 1000
    assert cfg.target == "labels"


def test_infer_config_from_run_picks_text_family_from_model_version():
    results = RunResults(run_dir=Path("run"), checkpoint_path=Path("run/trainval"),
                         cli_arguments={"modality": "text", "model_version": "gpt2", "mode": "cls"})
    assert infer_config_from_run(results, data=Path("new.csv"), out=Path("out")).model_str == "gpt2"

    results.cli_arguments["model_version"] = "BERT"
    assert infer_config_from_run(results, data=Path("new.csv"), out=Path("out")).model_str == "BERT"


def test_build_infer_argv_images_includes_model_build_settings():
    cfg = InferenceConfig(modality="images", checkpoint_path=Path("ckpt.pth"), data=Path("new"), out=Path("out"),
                          model_version="resnet50", num_classes=3, added_layers=2, embed_size=512,
                          freeze_backbone=True, use_peft=True, mode="cls")
    argv = build_infer_argv(cfg)
    assert argv[argv.index("--model_version") + 1] == "resnet50"
    assert argv[argv.index("--num_classes") + 1] == "3"
    assert "--freeze-backbone" in argv
    assert "--use-peft" in argv
    assert "--num_frames" not in argv  # images, not video


def test_build_infer_argv_video_includes_frame_settings():
    cfg = InferenceConfig(modality="video", checkpoint_path=Path("ckpt.pth"), data=Path("new"), out=Path("out"),
                          model_version="r3d_18", num_frames=16, pooling="attention", mode="cls")
    argv = build_infer_argv(cfg)
    assert argv[argv.index("--num_frames") + 1] == "16"
    assert argv[argv.index("--pooling") + 1] == "attention"


def test_build_infer_argv_tabular_gandalf_vs_tabpfn():
    gandalf_cfg = InferenceConfig(modality="tabular", checkpoint_path=Path("GANDALF_model"), data=Path("new.csv"),
                                  out=Path("out"), model_version="gandalf", target="price", mode="reg")
    argv = build_infer_argv(gandalf_cfg)
    assert argv[argv.index("--target") + 1] == "price"
    assert "--finetuning-mode" not in argv

    tabpfn_cfg = InferenceConfig(modality="tabular", checkpoint_path=Path("model.tabpfn_fit"), data=Path("new.csv"),
                                 out=Path("out"), model_version="tabpfn", finetuning_mode=True, mode="cls")
    argv = build_infer_argv(tabpfn_cfg)
    assert "--finetuning-mode" in argv


def test_build_infer_argv_timeseries_includes_history_paths():
    history = [Path("run/data_splits/train_split.parquet"), Path("run/data_splits/val_split.parquet")]
    cfg = InferenceConfig(modality="timeseries", checkpoint_path=Path("model.ckpt"), data=Path("new.csv"),
                          out=Path("out"), time_idx_column="t", target="value", mode="reg",
                          history_df_paths=history)
    argv = build_infer_argv(cfg)
    hist_idx = argv.index("--history_df")
    assert argv[hist_idx + 1: hist_idx + 3] == [str(p) for p in history]


def test_build_infer_argv_requires_checkpoint_data_and_out():
    with pytest.raises(ValueError):
        build_infer_argv(InferenceConfig(checkpoint_path=None, data=Path("new"), out=Path("out")))


def test_validate_infer_config_flags_missing_run_and_data_and_out():
    issues = validate_infer_config(InferenceConfig())
    assert any("run" in issue.lower() for issue in issues)
    assert any("data" in issue.lower() for issue in issues)
    assert any("output" in issue.lower() for issue in issues)


def test_validate_infer_config_flags_missing_timeseries_history(tmp_path):
    data = tmp_path / "new.csv"
    data.write_text("t,value\n1,2\n")
    cfg = InferenceConfig(run_dir=tmp_path, checkpoint_path=tmp_path / "model.ckpt", data=data,
                          out=tmp_path / "out", modality="timeseries", history_df_paths=[])
    issues = validate_infer_config(cfg)
    assert any("history" in issue.lower() for issue in issues)


def test_find_history_paths_locates_train_and_val_splits(tmp_path):
    splits_dir = tmp_path / "data_splits_20260101_0000"
    splits_dir.mkdir()
    pd.DataFrame({"a": [1]}).to_parquet(splits_dir / "train_split.parquet")
    pd.DataFrame({"a": [2]}).to_parquet(splits_dir / "val_split.parquet")
    paths = find_history_paths(tmp_path)
    assert len(paths) == 2
    assert any(p.name == "train_split.parquet" for p in paths)
    assert any(p.name == "val_split.parquet" for p in paths)


def test_find_history_paths_empty_when_no_splits(tmp_path):
    assert find_history_paths(tmp_path) == []


def test_read_inference_results_parses_predictions_and_metrics(tmp_path):
    pd.DataFrame({"sample_id": [0, 1], "predicted_label": ["cat", "dog"]}).to_csv(
        tmp_path / "predictions.csv", index=False)
    (tmp_path / "inference_metrics.json").write_text(json.dumps({"accuracy": 0.75}))
    results = read_inference_results(tmp_path)
    assert len(results.predictions) == 2
    assert results.metrics == {"accuracy": 0.75}
    assert results.warnings == []


def test_read_inference_results_missing_files_are_none(tmp_path):
    results = read_inference_results(tmp_path)
    assert results.predictions is None
    assert results.metrics is None

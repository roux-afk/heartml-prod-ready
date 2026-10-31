import json
import math
import plistlib
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from heart_disease import launchd, predict, retrain
from heart_disease.config import ALL_COLUMNS, FEATURE_COLUMNS
from heart_disease.models import build_pipeline
from heart_disease.registry import ActiveModel, ModelRegistry, atomic_json, read_json, smoke_check
from heart_disease.retrain_data import (
    InsufficientDataError,
    load_dataset,
    split_dataset,
    validate_numeric,
)
from heart_disease.retrain_settings import RetrainSettings, load_settings


def observations(count: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    target = np.arange(count) % 2
    return pd.DataFrame(
        {
            "age": rng.integers(35, 75, count),
            "sex": rng.integers(0, 2, count),
            "cp": 1 + target * 3,
            "trestbps": rng.normal(130, 5, count),
            "chol": 180 + target * 120 + rng.normal(0, 3, count),
            "fbs": rng.integers(0, 2, count),
            "restecg": rng.integers(0, 3, count),
            "thalach": rng.integers(100, 180, count),
            "exang": target,
            "oldpeak": target * 3 + rng.uniform(0, 0.1, count),
            "slope": 1 + target,
            "ca": target,
            "thal": 3 + target * 4,
            "target": target,
        }
    )


@pytest.fixture
def settings(tmp_path: Path) -> RetrainSettings:
    raw = tmp_path / "raw.csv"
    observations(120, 7).to_csv(raw, index=False, header=False)
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    return RetrainSettings(
        raw_path=raw,
        incoming_dir=incoming,
        store_dir=tmp_path / "models",
        models=("naive_bayes",),
        min_new_rows=20,
        min_eval_per_class=3,
        cv_folds=3,
    )


def write_config(settings: RetrainSettings, path: Path) -> None:
    path.write_text(
        "[retraining]\n"
        f"raw_path = {json.dumps(str(settings.raw_path))}\n"
        f"incoming_dir = {json.dumps(str(settings.incoming_dir))}\n"
        f"store_dir = {json.dumps(str(settings.store_dir))}\n"
        'models = ["naive_bayes"]\nmin_new_rows = 20\nmin_eval_per_class = 3\ncv_folds = 3\n',
        encoding="utf-8",
    )


def add_batch(settings: RetrainSettings) -> None:
    batch = observations(80, 99)
    batch["patient_id"] = [f"patient-{index // 2}" for index in range(len(batch))]
    batch.to_csv(settings.incoming_dir / "batch.csv", index=False)


def test_successful_deploy_consumes_data_and_inference_uses_active_version(
    settings: RetrainSettings,
) -> None:
    report = retrain.run_retraining(settings)
    assert report["status"] == "deployed"
    assert report["candidate_metrics"]["recall"] == 1
    active = ModelRegistry(settings.store_dir).active()
    assert active is not None and active["version"] == report["run_id"]
    run = settings.store_dir / "runs" / report["run_id"]
    development, evaluation = (
        pd.read_csv(run / "development.csv"),
        pd.read_csv(run / "evaluation.csv"),
    )
    assert not set(development.patient_id) & set(evaluation.patient_id)
    consumer = ActiveModel(settings.store_dir)
    assert len(consumer.predict(evaluation[FEATURE_COLUMNS])) == len(evaluation)
    assert consumer.version == report["run_id"]
    assert retrain.run_retraining(settings)["status"] == "skipped"


def test_rejected_candidate_preserves_active_model_and_never_reuses_holdout(
    settings: RetrainSettings,
) -> None:
    first = retrain.run_retraining(settings)
    assert first["status"] == "deployed"
    before = (settings.store_dir / "active.json").read_bytes()
    previous_holdout = set(retrain.read_history(settings.store_dir)["heldout_patients"])
    add_batch(settings)
    second = retrain.run_retraining(settings)
    assert second["status"] == "rejected"  # Both models are perfect: no improvement.
    assert (settings.store_dir / "active.json").read_bytes() == before
    run = settings.store_dir / "runs" / second["run_id"]
    development, evaluation = (
        pd.read_csv(run / "development.csv"),
        pd.read_csv(run / "evaluation.csv"),
    )
    assert not previous_holdout & set(development.patient_id)
    assert not previous_holdout & set(evaluation.patient_id)
    assert not set(development.patient_id) & set(evaluation.patient_id)
    assert retrain.run_retraining(settings)["status"] == "skipped"


def test_failed_training_and_bad_data_do_not_replace_active_model(
    settings: RetrainSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert retrain.run_retraining(settings)["status"] == "deployed"
    before = (settings.store_dir / "active.json").read_bytes()
    add_batch(settings)

    def fail(*args: object) -> None:
        raise RuntimeError("training failed")

    monkeypatch.setattr(retrain, "select_candidate", fail)
    assert retrain.run_retraining(settings)["status"] == "failed"
    assert (settings.store_dir / "active.json").read_bytes() == before
    (settings.incoming_dir / "bad.csv").write_text("wrong,schema\n1,2\n")
    report = retrain.run_retraining(settings)
    assert report["status"] == "failed" and report["stage"] == "data_validation"
    assert (settings.store_dir / "active.json").read_bytes() == before


def test_quality_gate_checks_thresholds_regression_and_missing_metrics(
    settings: RetrainSettings,
) -> None:
    perfect = {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    lower = {"precision": 0.9, "recall": 0.95, "f1": 0.95}
    assert retrain.quality_reasons(perfect, None, settings) == []
    assert retrain.quality_reasons(perfect, lower, settings) == []
    assert retrain.quality_reasons(perfect, perfect, settings)
    assert retrain.quality_reasons(lower, perfect, settings)
    assert retrain.quality_reasons({"recall": math.nan}, None, settings)
    assert retrain.quality_reasons(perfect, {**perfect, "f1": math.inf}, settings)


def test_concurrent_writer_is_skipped(settings: RetrainSettings) -> None:
    with retrain.run_lock(settings.store_dir):
        assert retrain.run_retraining(settings)["status"] == "busy"


def test_registry_reload_rollback_integrity_and_failed_activation(
    settings: RetrainSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = observations(40, 8)
    features = frame[FEATURE_COLUMNS]
    pipeline = build_pipeline("naive_bayes").fit(features, frame.target)
    store = ModelRegistry(settings.store_dir)
    for name in ("first", "second"):
        store.register(name, pipeline, {}, features)
    store.promote("first")
    consumer = ActiveModel(settings.store_dir)
    consumer.predict(features)
    store.promote("second")
    consumer.predict(features)
    assert consumer.version == "second"
    assert store.rollback() == "first"
    consumer.predict(features)
    assert consumer.version == "first"
    before = store.active_path.read_bytes()

    def fail(self: ActiveModel, features: pd.DataFrame) -> None:
        raise RuntimeError("consumer activation failed")

    with monkeypatch.context() as patch:
        patch.setattr(ActiveModel, "predict", fail)
        with pytest.raises(RuntimeError, match="activation failed"):
            store.promote("second")
    assert store.active_path.read_bytes() == before
    with pytest.raises(FileExistsError):
        store.register("first", pipeline, {}, features)
    (store.version_dir("second") / "model.joblib").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        store.load("second")
    with pytest.raises(ValueError, match="version"):
        store.version_dir("../escape")


def test_first_failed_activation_removes_pointer(
    settings: RetrainSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = observations(40, 2)
    features = frame[FEATURE_COLUMNS]
    store = ModelRegistry(settings.store_dir)
    pipeline = build_pipeline("naive_bayes").fit(features, frame.target)
    store.register("first", pipeline, {}, features)
    with pytest.raises(ValueError, match="previous"):
        store.rollback()

    def fail(self: ActiveModel, features: pd.DataFrame) -> None:
        raise RuntimeError("activation failed")

    monkeypatch.setattr(ActiveModel, "predict", fail)
    with pytest.raises(RuntimeError):
        store.promote("first")
    assert store.active() is None


@pytest.mark.parametrize(
    "column,value",
    [
        ("target", np.nan),
        ("target", 7),
        ("sex", 3),
        ("chol", -1),
        ("oldpeak", math.inf),
        ("age", "unknown"),
    ],
)
def test_invalid_labels_and_measurements_are_rejected(column: str, value: Any) -> None:
    frame = observations(40, 8).astype(object)
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_numeric(frame, 0.05)


def test_missing_values_duplicates_and_patient_ids(settings: RetrainSettings) -> None:
    frame = observations(40, 1)
    frame.loc[0, "ca"] = np.nan
    assert validate_numeric(frame, 0.05).ca.isna().sum() == 1
    frame["ca"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        validate_numeric(frame, 0.05)
    incoming = observations(40, 2)
    incoming["patient_id"] = "same-patient"
    pd.concat([incoming, incoming]).to_csv(settings.incoming_dir / "batch.csv", index=False)
    loaded, _ = load_dataset(settings)
    assert len(loaded) == 160
    incoming.loc[0, "patient_id"] = " "
    incoming.to_csv(settings.incoming_dir / "batch.csv", index=False)
    with pytest.raises(ValueError, match="patient_id"):
        load_dataset(settings)


def test_conflicting_labels_and_patient_identities(settings: RetrainSettings) -> None:
    incoming = observations(40, 2)
    incoming["patient_id"] = "one"
    duplicate = incoming.iloc[[0]].copy()
    duplicate["target"] = 1 - duplicate.target
    pd.concat([incoming, duplicate]).to_csv(settings.incoming_dir / "batch.csv", index=False)
    with pytest.raises(ValueError, match="target labels"):
        load_dataset(settings)
    duplicate["target"] = incoming.iloc[0].target
    duplicate["patient_id"] = "other"
    pd.concat([incoming, duplicate]).to_csv(settings.incoming_dir / "batch.csv", index=False)
    with pytest.raises(ValueError, match="patient identities"):
        load_dataset(settings)


def test_insufficient_fresh_patients_cannot_be_evaluation(settings: RetrainSettings) -> None:
    frame, _ = load_dataset(settings)
    history = {"seen_records": [], "known_patients": list(frame.patient_id), "heldout_patients": []}
    with pytest.raises(InsufficientDataError, match="Fresh"):
        split_dataset(frame, history, settings)
    with pytest.raises(InsufficientDataError, match="coverage"):
        split_dataset(frame, {}, replace(settings, min_eval_per_class=1000))


def test_corrupt_or_missing_history_is_not_reset(settings: RetrainSettings) -> None:
    atomic_json(settings.store_dir / "active.json", {"version": "unknown"})
    with pytest.raises(ValueError, match="lineage"):
        retrain.read_history(settings.store_dir)
    atomic_json(settings.store_dir / "history.json", {"unexpected": []})
    with pytest.raises(ValueError, match="history"):
        retrain.read_history(settings.store_dir)
    (settings.store_dir / "history.json").write_text("[]")
    with pytest.raises(ValueError, match="object"):
        read_json(settings.store_dir / "history.json")


@pytest.mark.parametrize(
    "values",
    [
        {"min_f1": math.nan},
        {"min_f1": -0.1},
        {"min_f1": "0.5"},
        {"cv_folds": 1},
        {"min_new_rows": True},
        {"seed": -1},
        {"models": ()},
        {"models": ("unknown",)},
        {"test_size": 1.0},
        {"min_improvement": 0},
    ],
)
def test_invalid_settings(values: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        RetrainSettings(**values)


def test_config_and_prediction_cli(settings: RetrainSettings, tmp_path: Path) -> None:
    config = tmp_path / "retraining.toml"
    write_config(settings, config)
    assert load_settings(config) == settings
    retrain.main(["--config", str(config)])
    source, output = tmp_path / "features.csv", tmp_path / "predictions.csv"
    observations(10, 12)[FEATURE_COLUMNS].to_csv(source, index=False)
    predict.main(["--config", str(config), "--input", str(source), "--output", str(output)])
    predictions = pd.read_csv(output)
    assert len(predictions) == 10 and set(predictions.prediction) <= {0, 1}
    assert predictions.model_version.eq(predictions.model_version.iloc[0]).all()
    with pytest.raises(FileNotFoundError, match="No active"):
        ActiveModel(tmp_path / "empty").predict(observations(10, 1)[FEATURE_COLUMNS])
    with pytest.raises(ValueError, match="Feature"):
        ActiveModel(settings.store_dir).predict(pd.DataFrame({"wrong": [1]}))
    config.write_text("[retraining]\nunknown = 1\n")
    with pytest.raises(ValueError, match="Unknown"):
        load_settings(config)
    config.write_text("[other]\n")
    with pytest.raises(ValueError, match="only"):
        load_settings(config)


def test_smoke_and_launchd_generation(settings: RetrainSettings, tmp_path: Path) -> None:
    config = tmp_path / "retraining.toml"
    write_config(settings, config)
    definition = launchd.scheduler_definition(config)
    assert definition["StartCalendarInterval"] == {"Day": 1, "Hour": 3, "Minute": 0}
    assert definition["WatchPaths"] == [str(settings.incoming_dir)]
    assert str(config) in definition["ProgramArguments"]
    launchd.main(["--config", str(config)])
    output = next((tmp_path / "ops/local").glob("*.plist"))
    assert plistlib.loads(output.read_bytes())["Label"] == definition["Label"]
    with pytest.raises(ValueError, match="nonempty"):
        smoke_check(build_pipeline("naive_bayes"), pd.DataFrame(columns=ALL_COLUMNS))


def test_cli_failed_run_has_nonzero_exit(settings: RetrainSettings, tmp_path: Path) -> None:
    config = tmp_path / "retraining.toml"
    write_config(settings, config)
    settings.raw_path.write_text("bad,data\n")
    with pytest.raises(SystemExit) as error:
        retrain.main(["--config", str(config)])
    assert error.value.code == 1


def test_json_write_failure_preserves_old_document(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    atomic_json(path, {"old": True})
    with pytest.raises(ValueError):
        atomic_json(path, {"invalid": math.nan})
    assert read_json(path) == {"old": True}
    assert list(tmp_path.iterdir()) == [path]


def test_nonfinite_metrics_are_rejected_and_reported_as_null(
    settings: RetrainSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        retrain,
        "evaluate",
        lambda *args, **kwargs: {
            "f1": math.nan,
            "recall": 0.95,
            "precision": 0.9,
        },
    )
    report = retrain.run_retraining(settings)
    assert report["status"] == "rejected"
    assert read_json(settings.store_dir / "last_run.json")["candidate_metrics"]["f1"] is None
    assert ModelRegistry(settings.store_dir).active() is None


def test_explicit_rollback_cli(settings: RetrainSettings, tmp_path: Path) -> None:
    config = tmp_path / "retraining.toml"
    write_config(settings, config)
    first = retrain.run_retraining(settings)
    store = ModelRegistry(settings.store_dir)
    store.register("second", store.load(first["run_id"]), {}, observations(10, 7)[FEATURE_COLUMNS])
    store.promote("second")
    retrain.main(["--config", str(config), "--rollback"])
    active = store.active()
    assert active is not None and active["version"] == first["run_id"]
    assert read_json(settings.store_dir / "last_rollback.json")["status"] == "rolled_back"


def test_scheduler_installation_uses_user_launchagent(
    settings: RetrainSettings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "retraining.toml"
    write_config(settings, config)
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> None:
        calls.append(command)

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.setattr(subprocess, "run", fake_run)
    launchd.main(["--config", str(config), "--install"])
    installed = next((tmp_path / "home/Library/LaunchAgents").glob("*.plist"))
    assert plistlib.loads(installed.read_bytes())["RunAtLoad"] is True
    assert calls[-1][-1] == str(installed) and calls[-1][1] == "bootstrap"
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="macOS"):
        launchd.main(["--config", str(config), "--install"])


def test_smoke_rejects_invalid_predictions_and_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = observations(20, 5)
    features = frame[FEATURE_COLUMNS]
    pipeline = build_pipeline("naive_bayes").fit(features, frame.target)
    with monkeypatch.context() as patch:
        patch.setattr(pipeline, "predict", lambda batch: np.full(len(batch), 3))
        with pytest.raises(ValueError, match="classes"):
            smoke_check(pipeline, features)
    monkeypatch.setattr(pipeline, "predict_proba", lambda batch: np.full((len(batch), 2), np.nan))
    with pytest.raises(ValueError, match="probabilities"):
        smoke_check(pipeline, features)

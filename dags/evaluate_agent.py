"""Airflow DAG for SWE-bench agent evaluation with durable run artifacts.

Pipeline: prepare_run -> run_agent -> run_eval -> summarize_and_log.

Every trigger materializes a self-contained ``runs/<run-id>/`` directory holding
the configuration, agent trajectories, predictions, evaluation logs/reports,
aggregate metrics, and a manifest that points at the important files.
"""

from __future__ import annotations

import json
import os
import subprocess
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from airflow.decorators import dag, task
from airflow.models.param import Param

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
RUNS_ROOT = PROJECT_ROOT / "runs"

SUBSET_TO_DATASET = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
    "full": "princeton-nlp/SWE-bench",
}
AGENT_TRAJECTORIES_DIR = "trajectories"
EVAL_LOGS_DIR = "logs"
EVAL_REPORTS_DIR = "reports"


def _sanitize_model(model: str) -> str:
    return model.replace("/", "__")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_run_config(params: dict[str, Any]) -> dict[str, Any]:
    """Normalize Airflow params into a serializable run config."""
    raw_run_id = (params.get("run_id") or "").strip()
    run_id = raw_run_id or _utc_stamp()
    model = (params.get("model") or os.environ.get("MSWEA_MODEL") or "").strip()
    if not model:
        raise ValueError("model must be provided via Airflow params or MSWEA_MODEL")
    return {
        "run_id": run_id,
        "split": params["split"],
        "subset": params["subset"],
        "workers": int(params["workers"]),
        "model": model,
        "task_slice": (params.get("task_slice") or "").strip(),
        "object_storage_bucket": (
            params.get("object_storage_bucket") or os.environ.get("OBJECT_STORAGE_BUCKET") or ""
        ).strip(),
        "object_storage_prefix": (
            params.get("object_storage_prefix") or os.environ.get("OBJECT_STORAGE_PREFIX") or ""
        ).strip(),
        "object_storage_endpoint_url": (
            params.get("object_storage_endpoint_url")
            or os.environ.get("OBJECT_STORAGE_ENDPOINT_URL")
            or os.environ.get("AWS_ENDPOINT_URL_S3")
            or ""
        ).strip(),
        "object_storage_region": (
            params.get("object_storage_region")
            or os.environ.get("AWS_DEFAULT_REGION")
            or os.environ.get("AWS_REGION")
            or ""
        ).strip(),
        "dataset_name": SUBSET_TO_DATASET.get(
            params["subset"], f"princeton-nlp/SWE-bench_{params['subset'].capitalize()}"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def prepare_run_dir(run_config: dict[str, Any]) -> Path:
    """Create runs/<run-id> with the final durable layout and write config.json."""
    run_dir = RUNS_ROOT / run_config["run_id"]
    (run_dir / "run-agent" / AGENT_TRAJECTORIES_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / "run-eval" / EVAL_LOGS_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / "run-eval" / EVAL_REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2))
    return run_dir


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _move_run_agent_outputs(agent_dir: Path) -> None:
    """Normalize mini-swe-agent outputs into preds.json + trajectories/."""
    trajectories_dir = agent_dir / AGENT_TRAJECTORIES_DIR
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    for item in list(agent_dir.iterdir()):
        if item.name in {"preds.json", AGENT_TRAJECTORIES_DIR}:
            continue
        shutil.move(str(item), str(trajectories_dir / item.name))


def _copy_eval_reports(eval_dir: Path) -> Path:
    """Copy SWE-bench summary JSON files into run-eval/reports/."""
    reports_dir = eval_dir / EVAL_REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)

    for candidate in eval_dir.glob("*.json"):
        shutil.copy2(candidate, reports_dir / candidate.name)

    return reports_dir


def _build_object_storage_context(run_config: dict[str, Any]) -> dict[str, Any] | None:
    bucket = (run_config.get("object_storage_bucket") or "").strip()
    if not bucket:
        return None

    prefix = (run_config.get("object_storage_prefix") or "").strip().strip("/")
    key_prefix_parts = [part for part in [prefix, run_config["run_id"]] if part]
    key_prefix = "/".join(key_prefix_parts)

    return {
        "bucket": bucket,
        "key_prefix": key_prefix,
        "artifact_uri": f"s3://{bucket}/{key_prefix}" if key_prefix else f"s3://{bucket}",
        "endpoint_url": (run_config.get("object_storage_endpoint_url") or "").strip() or None,
        "region": (run_config.get("object_storage_region") or "").strip() or None,
    }


def _make_s3_client(storage_ctx: dict[str, Any]):
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required for Object Storage uploads. Add it to the environment first."
        ) from exc

    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip()
    session_token = os.environ.get("AWS_SESSION_TOKEN", "").strip() or None
    if not access_key or not secret_key:
        raise ValueError(
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required when Object Storage is enabled."
        )

    return boto3.client(
        "s3",
        endpoint_url=storage_ctx["endpoint_url"],
        region_name=storage_ctx["region"],
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        aws_session_token=session_token,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def _upload_run_tree_to_s3(
    run_dir: Path,
    storage_ctx: dict[str, Any],
    *,
    skip_relative_paths: set[str] | None = None,
) -> None:
    s3_client = _make_s3_client(storage_ctx)
    skip_relative_paths = skip_relative_paths or set()

    for file_path in sorted(run_dir.rglob("*")):
        if not file_path.is_file():
            continue
        relative_path = file_path.relative_to(run_dir).as_posix()
        if relative_path in skip_relative_paths:
            continue
        object_key = "/".join(
            part for part in [storage_ctx["key_prefix"], relative_path] if part
        )
        s3_client.upload_file(str(file_path), storage_ctx["bucket"], object_key)


def _upload_file_to_s3(run_dir: Path, storage_ctx: dict[str, Any], relative_path: str) -> None:
    s3_client = _make_s3_client(storage_ctx)
    file_path = run_dir / relative_path
    object_key = "/".join(part for part in [storage_ctx["key_prefix"], relative_path] if part)
    s3_client.upload_file(str(file_path), storage_ctx["bucket"], object_key)


def run_agent_batch(run_config: dict[str, Any], run_dir: Path) -> Path:
    """Invoke mini-swe-agent batch; expect preds.json + trajectories under run-agent/."""
    agent_dir = run_dir / "run-agent"

    cmd = [
        "uv",
        "run",
        "mini-extra",
        "swebench",
        "--subset", run_config["subset"],
        "--split", run_config["split"],
        "--model", run_config["model"],
        "--workers", str(run_config["workers"]),
        "-o", str(agent_dir),
    ]
    if run_config.get("task_slice"):
        cmd += ["--slice", run_config["task_slice"]]

    benchmark_yaml = (
        PROJECT_ROOT
        / "mini-swe-agent" / "src" / "minisweagent" / "config" / "benchmarks" / "swebench.yaml"
    )
    if benchmark_yaml.exists():
        cmd += ["--config", str(benchmark_yaml)]

    env = {**os.environ, "MSWEA_COST_TRACKING": "ignore_errors"}
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)
    _move_run_agent_outputs(agent_dir)

    return agent_dir / "preds.json"


def run_swebench_eval(run_config: dict[str, Any], preds_path: Path, run_dir: Path) -> Path:
    """Invoke SWE-bench harness with cwd=run-eval/ so logs land inside the run folder."""
    eval_dir = run_dir / "run-eval"
    eval_run_id = run_config["split"]

    cmd = [
        "uv", "run", "python", "-m", "swebench.harness.run_evaluation",
        "--dataset_name", run_config["dataset_name"],
        "--predictions_path", str(preds_path),
        "--max_workers", str(run_config["workers"]),
        "--run_id", eval_run_id,
    ]
    subprocess.run(cmd, cwd=eval_dir, check=True)
    _copy_eval_reports(eval_dir)
    return eval_dir


def collect_metrics(eval_dir: Path) -> dict[str, Any]:
    """Parse the SWE-bench aggregate report JSON from reports/ or eval root."""
    summary: dict[str, Any] = {}
    report_candidates = list((eval_dir / EVAL_REPORTS_DIR).glob("*.json")) + list(
        eval_dir.glob("*.json")
    )
    for candidate in sorted(report_candidates):
        try:
            data = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and "submitted_instances" in data:
            summary = data
            break

    if not summary:
        return {}

    submitted = summary.get("submitted_instances", 0) or 0
    resolved = summary.get("resolved_instances", 0) or 0
    metrics = {
        "total_instances": summary.get("total_instances", 0),
        "submitted_instances": submitted,
        "completed_instances": summary.get("completed_instances", 0),
        "resolved_instances": resolved,
        "unresolved_instances": summary.get("unresolved_instances", 0),
        "empty_patch_instances": summary.get("empty_patch_instances", 0),
        "error_instances": summary.get("error_instances", 0),
        "resolution_rate": (resolved / submitted) if submitted else 0.0,
    }
    return metrics


def log_mlflow_run(
    run_config: dict[str, Any], metrics: dict[str, Any], artifact_uri: str
) -> None:
    """Log params, metrics, and the artifact location to MLflow. Soft-fail if missing."""
    try:
        import mlflow
    except ImportError:
        print("[mlflow] python package not installed; skipping MLflow logging.")
        return

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or f"file:{PROJECT_ROOT / 'mlruns'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "mini-swe-bench"))

    with mlflow.start_run(run_name=run_config["run_id"]):
        for key, value in run_config.items():
            mlflow.log_param(key, value)
        mlflow.log_param("artifact_uri", artifact_uri)
        for key, value in metrics.items():
            try:
                mlflow.log_metric(key, float(value))
            except (TypeError, ValueError):
                continue


@dag(
    dag_id="evaluate_agent",
    description="Run mini-swe-agent on a SWE-bench subset and evaluate the patches.",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlops", "swebench", "mini-swe-agent"],
    params={
        "split": Param("test", type="string", description="SWE-bench split, e.g. test/dev"),
        "subset": Param(
            "verified",
            type="string",
            enum=["lite", "verified", "full"],
            description="SWE-bench subset to evaluate against.",
        ),
        "workers": Param(1, type="integer", minimum=1, description="Parallel workers."),
        "model": Param(
            "",
            type="string",
            description="LiteLLM-style model id for mini-swe-agent. Set via param or MSWEA_MODEL.",
        ),
        "task_slice": Param(
            "",
            type="string",
            description="Python-style slice over the dataset, e.g. '0:10'. Empty = all.",
        ),
        "object_storage_bucket": Param(
            "",
            type=["null", "string"],
            description="Optional S3/Object Storage bucket name for run artifacts.",
        ),
        "run_id": Param(
            "",
            type="string",
            description="Folder name under runs/. Empty = auto UTC timestamp.",
        ),
    },
)
def evaluate_agent_dag():
    @task
    def prepare_run(**ctx: Any) -> dict[str, Any]:
        run_config = build_run_config(ctx["params"])
        run_dir = prepare_run_dir(run_config)
        return {"run_config": run_config, "run_dir": str(run_dir)}

    @task
    def run_agent(state: dict[str, Any]) -> dict[str, Any]:
        preds_path = run_agent_batch(state["run_config"], Path(state["run_dir"]))
        return {**state, "preds_path": str(preds_path)}

    @task
    def run_eval(state: dict[str, Any]) -> dict[str, Any]:
        eval_dir = run_swebench_eval(
            state["run_config"], Path(state["preds_path"]), Path(state["run_dir"])
        )
        return {**state, "eval_dir": str(eval_dir)}

    @task
    def summarize_and_log(state: dict[str, Any]) -> dict[str, Any]:
        run_config = state["run_config"]
        run_dir = Path(state["run_dir"])
        eval_dir = Path(state["eval_dir"])

        metrics = collect_metrics(eval_dir)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        eval_reports_dir = eval_dir / EVAL_REPORTS_DIR
        storage_ctx = _build_object_storage_context(run_config)
        artifact_uri = str(run_dir)

        manifest = {
            "schema_version": 1,
            "run_id": run_config["run_id"],
            "created_at": run_config["created_at"],
            "dataset_name": run_config["dataset_name"],
            "split": run_config["split"],
            "subset": run_config["subset"],
            "workers": run_config["workers"],
            "model": run_config["model"],
            "task_slice": run_config["task_slice"],
            "local_artifact_uri": str(run_dir),
            "artifact_uri": artifact_uri,
            "config": "config.json",
            "run_agent": {
                "predictions": "run-agent/preds.json",
                "trajectories_dir": f"run-agent/{AGENT_TRAJECTORIES_DIR}",
            },
            "run_eval": {
                "logs_dir": f"run-eval/{EVAL_LOGS_DIR}",
                "reports_dir": f"run-eval/{EVAL_REPORTS_DIR}",
                "report_files": [
                    str(path.relative_to(run_dir)) for path in sorted(eval_reports_dir.glob("*.json"))
                ],
            },
            "metrics": "metrics.json",
        }
        if storage_ctx is not None:
            manifest["object_storage"] = {
                "bucket": storage_ctx["bucket"],
                "prefix": storage_ctx["key_prefix"],
                "endpoint_url": storage_ctx["endpoint_url"],
                "region": storage_ctx["region"],
                "artifact_uri": storage_ctx["artifact_uri"],
            }
        _write_json(run_dir / "manifest.json", manifest)

        if storage_ctx is not None:
            _upload_run_tree_to_s3(run_dir, storage_ctx, skip_relative_paths={"manifest.json"})
            artifact_uri = storage_ctx["artifact_uri"]
            manifest["artifact_uri"] = artifact_uri
            _write_json(run_dir / "manifest.json", manifest)
            _upload_file_to_s3(run_dir, storage_ctx, "manifest.json")

        log_mlflow_run(run_config, metrics, artifact_uri)
        return {"metrics": metrics, "manifest": str(run_dir / "manifest.json")}

    summarize_and_log(run_eval(run_agent(prepare_run())))


evaluate_agent_dag()

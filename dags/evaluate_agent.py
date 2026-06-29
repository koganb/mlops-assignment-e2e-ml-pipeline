"""Phase 1 DAG: run mini-swe-agent on a SWE-bench subset, evaluate, log to MLflow.

Pipeline: prepare_run -> run_agent -> run_eval -> summarize_and_log.

Every trigger materializes a self-contained ``runs/<run-id>/`` directory holding
the configuration, agent trajectories, predictions, evaluation logs/reports,
aggregate metrics, and a manifest that points at the important files.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from airflow.decorators import dag, task
from airflow.models.param import Param

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"

SUBSET_TO_DATASET = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
    "full": "princeton-nlp/SWE-bench",
}


def _sanitize_model(model: str) -> str:
    return model.replace("/", "__")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_run_config(params: dict[str, Any]) -> dict[str, Any]:
    """Normalize Airflow params into a serializable run config."""
    raw_run_id = (params.get("run_id") or "").strip()
    run_id = raw_run_id or _utc_stamp()
    return {
        "run_id": run_id,
        "split": params["split"],
        "subset": params["subset"],
        "workers": int(params["workers"]),
        "model": params["model"],
        "task_slice": (params.get("task_slice") or "").strip(),
        "dataset_name": SUBSET_TO_DATASET.get(
            params["subset"], f"princeton-nlp/SWE-bench_{params['subset'].capitalize()}"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def prepare_run_dir(run_config: dict[str, Any]) -> Path:
    """Create runs/<run-id>/{run-agent,run-eval}/ and write config.json."""
    run_dir = RUNS_ROOT / run_config["run_id"]
    (run_dir / "run-agent").mkdir(parents=True, exist_ok=True)
    (run_dir / "run-eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2))
    return run_dir


def run_agent_batch(run_config: dict[str, Any], run_dir: Path) -> Path:
    """Invoke mini-swe-agent batch; expect preds.json + trajectories under run-agent/."""
    agent_dir = run_dir / "run-agent"

    cmd = [
        "uv", "run", "mini-extra", "swebench",
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
    return eval_dir


def collect_metrics(eval_dir: Path) -> dict[str, Any]:
    """Parse the SWE-bench aggregate report JSON at the root of eval_dir."""
    summary: dict[str, Any] = {}
    for candidate in sorted(eval_dir.glob("*.json")):
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
            "nebius/moonshotai/Kimi-K2.6",
            type="string",
            description="LiteLLM-style model id for mini-swe-agent.",
        ),
        "task_slice": Param(
            "0:3",
            type="string",
            description="Python-style slice over the dataset, e.g. '0:10'. Empty = all.",
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

        manifest = {
            "run_id": run_config["run_id"],
            "created_at": run_config["created_at"],
            "model": run_config["model"],
            "config": "config.json",
            "predictions": "run-agent/preds.json",
            "trajectories_dir": "run-agent",
            "eval_dir": "run-eval",
            "metrics": "metrics.json",
            "artifact_uri": str(run_dir),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        log_mlflow_run(run_config, metrics, str(run_dir))
        return {"metrics": metrics, "manifest": str(run_dir / "manifest.json")}

    summarize_and_log(run_eval(run_agent(prepare_run())))


evaluate_agent_dag()

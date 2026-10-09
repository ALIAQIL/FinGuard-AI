"""Airflow DAG for finite local batch jobs; streaming is intentionally separate."""

from datetime import datetime

from airflow.decorators import dag, task


@dag(
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["finguard", "fraud"],
)
def finguard_batch():
    @task
    def validate_input() -> str:
        from pathlib import Path

        path = Path("data/raw/creditcard.csv")
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        return str(path)

    @task
    def profile(path: str) -> str:
        from finguard.phase1.eda import profile as run_profile

        return str(run_profile(path, "reports/phase1", "local[2]"))

    @task
    def benchmark(path: str) -> str:
        from finguard.benchmark import run_benchmark

        result = run_benchmark(path, "reports/benchmark_ulb.json")
        return str(result["models"])

    source = validate_input()
    profile(source)
    benchmark(source)


finguard_batch()


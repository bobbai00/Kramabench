import argparse
import json
import os
import pandas as pd
import traceback

from benchmark import Benchmark
from benchmark.metrics import metric_factory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sut", type=str, default="BaselineLLMSystemGPT4oFewShot", help="The system under test.")
    parser.add_argument("--dataset_name", type=str, default="environment", help="Name of dataset.")
    parser.add_argument("--workload_filename", type=str, default="environment.json", help="Name of workload JSON file.")
    parser.add_argument("--result_directory", type=str, default="results", help="Directory to store benchmark results.")
    parser.add_argument("--task_fixtures", type=str, default="benchmark/fixtures", help="Directory containing task fixture files.")
    parser.add_argument("--project_root", type=str, default=os.getcwd(), help="Project root.")
    parser.add_argument("--use-cache", action="store_true", default=True, help="Use cached system outputs if available.")
    parser.add_argument("--verbose", action="store_true", default=False, help="Verbose logging.")
    args = parser.parse_args()

    system_name = args.sut
    result_root_dir = os.path.join(args.project_root, args.result_directory)
    task_fixture_dir = os.path.join(args.project_root, args.task_fixtures)
    use_cache = args.use_cache
    verbose = args.verbose
    project_root_dir = args.project_root
    dataset_name = args.dataset_name
    workload_filename = args.workload_filename
    # Setup output directory for system under test
    system_output_dir = os.path.join(project_root_dir, f"system_scratch/{system_name}")
    os.makedirs(system_output_dir, exist_ok=True)

    workload_name = os.path.basename(workload_filename)
    workload_path = os.path.join(project_root_dir, f"workload/{workload_filename}")
    system_result_dir = os.path.join(result_root_dir, system_name)
    os.makedirs(system_result_dir, exist_ok=True)
    result_path = os.path.join(system_result_dir, f"{workload_name}_results.json")
    measures_path = os.path.join(system_result_dir, f"{workload_name}_measures.csv")
    aggregated_results_path = os.path.join(result_root_dir, "aggregated_results.csv")

    benchmark = Benchmark(
        system_name=system_name,
        task_fixture_directory=task_fixture_dir,
        system_output_directory=system_output_dir,
        cache_system_output=use_cache,
        verbose=verbose,
    )

    if use_cache and os.path.exists(result_path):
        print(f"Loading cached results from {result_path}")
        with open(result_path) as f:
            results = json.load(f)
    else:
        print(f"Running benchmark on workload: {workload_name}")
        results, evaluation_results = benchmark.run_benchmark(
            dataset_directory=os.path.join(project_root_dir, f"data/{dataset_name}/input"),
            results_directory=system_result_dir,
            workload_path=workload_path,
            verbose=verbose
        )
        with open(result_path, "w") as f:
            json.dump(results, f)

    # Pretty printing results
    flat_measures = []
    for task_result in evaluation_results:
        for result in task_result:
            task_id = result.pop("task_id", None)
            for metric, value in result.items():
                flat_measures.append({
                    "sut": system_name,
                    "workload": workload_name,
                    "task_id": task_id,
                    "metric": metric,
                    "value": value
                })

    results_df = pd.DataFrame(flat_measures)
    results_df.to_csv(measures_path, index=False)

    if verbose:
        print(results_df)

    # Aggregate metrics
    print("Aggregating results...")
    workload_results = []
    for (workload, metric), group in results_df.groupby(["workload", "metric"]):
        mean = group["value"].mean()
        std = group["value"].std() if len(group) > 1 else 0
        workload_results.append({
            "sut": system_name,
            "workload": workload,
            "metric": metric,
            "value_mean": mean,
            "value_std": std,
            "value_support": len(group)
        })

    aggregated_df = pd.DataFrame(workload_results)

    # Update aggregated results file
    if os.path.exists(aggregated_results_path):
        old_aggregated_df = pd.read_csv(aggregated_results_path)
        old_aggregated_df = old_aggregated_df[
            ~((old_aggregated_df["sut"] == system_name) & (old_aggregated_df["workload"] == workload_name))
        ]
        aggregated_df = pd.concat([old_aggregated_df, aggregated_df])

    aggregated_df.to_csv(aggregated_results_path, index=False)

    print(f"Done. Aggregated results:")
    print(aggregated_df[aggregated_df["workload"] == workload_name])


if __name__ == "__main__":
    main()

import argparse
import os
import pandas as pd

from benchmark import Benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sut", type=str, default="BaselineLLMSystemGPT4oFewShot", help="The system under test.")
    parser.add_argument("--dataset_name", type=str, default="environment", help="Name of dataset.")
    parser.add_argument("--workload_filename", type=str, default="environment.json", help="Name of workload JSON file.")
    parser.add_argument("--result_directory", type=str, default="results", help="Directory to store benchmark results.")
    parser.add_argument("--task_fixtures", type=str, default="benchmark/fixtures", help="Directory containing task fixture files.")
    parser.add_argument("--project_root", type=str, default=os.getcwd(), help="Project root.")
    parser.add_argument("--use_system_cache", action="store_true", default=True, help="Use cached system outputs if available.")
    parser.add_argument("--cache_system_output", action="store_true", default=True, help="Cache system output.")
    parser.add_argument("--verbose", action="store_true", default=False, help="Verbose logging.")
    args = parser.parse_args()

    system_name = args.sut
    verbose = args.verbose

    project_root_dir = args.project_root
    result_root_dir = os.path.join(project_root_dir, args.result_directory)

    # Setup output (cache maintained by benchmark) and scratch directories for system under test
    system_result_dir = os.path.join(result_root_dir, system_name)
    workload_filename = args.workload_filename
    workload_name = os.path.basename(workload_filename)
    workload_path = os.path.join(project_root_dir, f"workload/{workload_filename}")
    system_output_dir = os.path.join(project_root_dir, f"system_scratch/{system_name}")
    os.makedirs(system_output_dir, exist_ok=True)
    os.makedirs(system_result_dir, exist_ok=True)

    # Setup benchmark evaluation util directory
    task_fixture_dir = os.path.join(project_root_dir, args.task_fixtures)
    measures_path = os.path.join(system_result_dir, f"{workload_name}_measures.csv")
    aggregated_results_path = os.path.join(result_root_dir, "aggregated_results.csv")

    benchmark = Benchmark(
        system_name=system_name,
        task_fixture_directory=task_fixture_dir,
        system_output_directory=system_output_dir,
        use_system_cache=args.use_system_cache,
        cache_system_output=args.cache_system_output,
        verbose=verbose,
    )

    print(f"Running benchmark on workload: {workload_name}")
    _, evaluation_results = benchmark.run_benchmark(
        dataset_directory=os.path.join(project_root_dir, f"data/{args.dataset_name}/input"),
        results_directory=system_result_dir,
        workload_path=workload_path,
        verbose=verbose
    )

    # Pretty printing evaluation_results
    flat_measures = []
    for task_result in evaluation_results:
        # TODO: Implement system planned subtask result evaluation
        task_id = task_result["task_id"]
        for metric, value in task_result.items():
            if metric == "task_id":
                continue
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

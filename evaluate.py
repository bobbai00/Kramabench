""" This script can be used to evaluate the results of a system run.
The systems can be called using the run_pipeline function in system.api with the following run types:
- baseline: single model name (str)
- cross_validation: list of model names to cross validate, and a final model to merge the results
- reflection: one model for reflection, and another model to critic and decide to accept or reject the reflection
"""

from __future__ import print_function
import json
import os
import argparse
import traceback
import re

import pandas as pd
from typing import List

from benchmark.metrics import Precision, Recall, F1, BleuScore, RougeScore, Success

METRICS = [Precision, Recall, F1, BleuScore, RougeScore, Success]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sut", default=None, help="The system to benchmark")
    parser.add_argument("--workload", default="queries/easy/demo.json", help="The json file containing the input queries")
    parser.add_argument("--result", default="./results", help="The root path where the results of the pipelines are stored.")
    parser.add_argument("--verbose", default=False, help="Whether to print filenames as they are processed")

    args = parser.parse_args()
    sut = args.sut
    workload = args.workload
    RESULT_DIR = args.result

    verbose = bool(args.verbose)
    workload_name = os.path.basename(workload)
    result_file = f"{RESULT_DIR}/{sut}/{workload_name}_measures.csv"

    with open(workload) as f:
        queries = json.load(f)

    result_path = f"{RESULT_DIR}/{sut}/{workload_name}"
    with open(result_path) as f:
        sut_answers = json.load(f)    


    workload_measures = []
    for idx, query in enumerate(queries):
        target = query
        predicted = sut_answers[idx]
        
        for metric in METRICS:
            try:
                value = metric(predicted, target)
            except Exception as e:
                print("Exception:", traceback.format_exc())
                if not verbose:
                    print("On query:", idx)
                value = 0

            dict_measures = {
                "workload": workload_name,
                "query_idx": idx,
                "metric": metric.name,
                "value": value
                }
            workload_measures.append(dict_measures)

    results_df = pd.DataFrame(workload_measures)
    results_df.to_csv(result_file, index=False)
    if verbose: 
        print(results_df)

    # Logic to aggregate the results
    # workload_results = [] 
    
if __name__ == "__main__":
    main()
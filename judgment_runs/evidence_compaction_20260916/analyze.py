# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the
# specific language governing permissions and limitations
# under the License.

"""Reproduce the single-pass report without dispatching or re-scoring tasks.

Read the committed per-task export and aggregate its original native metrics
using compute_scores_for_system. Only stdout is written. Recovery rows,
duplicate attempts and incomplete task sets are rejected, never selected.
"""

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parents[1]
sys.path.insert(0, str(HARNESS))

import pandas as pd
from compute_scores import SCORE_METRICS, compute_scores_for_system

ARMS = ('Native', 'DataOnly', 'FlowOnly', 'Combined')
DOMAINS = ('archeology', 'astronomy', 'biomedical', 'environment', 'legal', 'wildfire')
TOKEN_FIELDS = ('input_tokens', 'cached_tokens', 'output_tokens', 'reasoning_tokens')


def load_rows(path):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if row['dispatched'] not in ('true', 'false'):
            raise ValueError('dispatched must be true or false')
        row['dispatched'] = row['dispatched'] == 'true'
        row['score'] = float(row['score'])
        row['cost_usd'] = float(row['cost_usd']) if row['cost_usd'] else None
        for key in TOKEN_FIELDS:
            row[key] = int(row[key]) if row[key] else None
    return rows


def canonical_task_ids():
    return {task['id'] for domain in DOMAINS
            for task in json.loads((HARNESS / 'workload' / f'{domain}.json').read_text())}


def summarize(rows, expected_task_ids=None):
    expected = canonical_task_ids() if expected_task_ids is None else set(expected_task_ids)
    grouped, identities = defaultdict(list), set()
    for row in rows:
        if row['round'] != 'first':
            raise ValueError('Only original first attempts may enter this report')
        key = (row['configuration'], row['task_id'])
        if key in identities or key[0] not in ARMS:
            raise ValueError('Duplicate task or unknown configuration')
        identities.add(key)
        if row['metric'] not in SCORE_METRICS or not math.isfinite(row['score']):
            raise ValueError('Invalid native metric')
        cost = row['cost_usd']
        if cost is not None and (not math.isfinite(cost) or cost < 0):
            raise ValueError('Invalid recorded model cost')
        if not row['dispatched'] and (row['status'] != 'data_readiness_blocked' or row['score'] != 0):
            raise ValueError('Undispatched task is not a documented input blocker')
        grouped[key[0]].append(row)
    reports = []
    for key in ARMS:
        arm = grouped[key]
        if {row['task_id'] for row in arm} != expected:
            raise ValueError('Every arm must contain the same complete canonical task set')
        metrics = defaultdict(list)
        for row in arm:
            metrics[(row['task_id'].rsplit('-', 2)[0], row['metric'])].append(row['score'])
        frame = pd.DataFrame([dict(workload=domain, metric=metric, value_support=len(values),
                                   total_value_support=len(values), value_mean=sum(values)/len(values))
                              for (domain, metric), values in metrics.items()])
        dispatched = [row for row in arm if row['dispatched']]
        unknown = sum(row['cost_usd'] is None for row in dispatched)
        cost = round(sum(row['cost_usd'] for row in dispatched if row['cost_usd'] is not None), 6)
        reports.append(dict(
            configuration=key, tasks=len(arm), model_dispatches=len(dispatched),
            input_blocked=len(arm)-len(dispatched),
            exact_passes=sum(row['status'] == 'completed' and row['score'] >= 1 for row in arm),
            native_scores=compute_scores_for_system(frame, key).to_dict('records'),
            known_model_cost_usd=cost, model_attempts_with_unknown_cost=unknown,
            mean_cost_per_executed_task_usd=cost/len(dispatched) if dispatched and not unknown else None,
            **{field: sum(row[field] or 0 for row in dispatched) for field in TOKEN_FIELDS}))
    return dict(round='first', model='gpt-5.6-terra', reasoning='medium',
                scoring='Original KramaBench primary metrics; compute_scores_for_system aggregation',
                cost_basis='Recorded cache-aware model-token estimates; mean over dispatched tasks',
                native_baseline='Historical V1 batch+observe, not a matched V2 no-evidence control',
                retries_included=False, arms=reports)


if __name__ == '__main__':
    print(json.dumps(summarize(load_rows(HERE / 'first_pass_tasks.csv')), indent=2, allow_nan=False))

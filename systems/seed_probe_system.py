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

"""Seeded source roots: control vs seeded arms, 2026-09-20.

Every knob is the 2026-09-16 compact-evidence campaign's (imported verbatim
from the layout probe's `_FROZEN`), Data-only evidence, DELTA. The only
difference between the control and the seeded arm is `seed_sources`:

    control  the model discovers the task's files itself (scan-only steps)
    seeded   the service materializes one scan per listed file, observed,
             before the first model call; the model's first step already
             sees every source's schema, sample rows and data evidence.

A third arm combines seeding with the retained-observation LATEST layout
(`message_layout="opBlock"`) so both of tonight's changes are measured on
the same tasks.
"""

import os

from .dataflow_system import DataflowSystem
from .layout_probe_system import _FROZEN


class _SeedProbeBase(DataflowSystem):
    _CONTEXT_MODE = "delta"
    _MESSAGE_LAYOUT = None
    _SEED = False
    _NAME = "_SeedProbeBase"

    def __init__(self, verbose: bool = False, *args, **kwargs):
        kwargs.setdefault(
            "agent_service_endpoint", os.environ.get("SEED_PROBE_AGENT_ENDPOINT", "http://localhost:3061")
        )
        super().__init__(
            context_mode=self._CONTEXT_MODE,
            message_layout=self._MESSAGE_LAYOUT,
            data_evidence=True,
            flow_evidence=False,
            seed_sources=self._SEED,
            name=self._NAME,
            verbose=verbose,
            *args,
            **_FROZEN,
            **kwargs,
        )


class DataflowSystemTerraSeedControlDeltaDataOnly20260920(_SeedProbeBase):
    """Control: campaign Data-only protocol, no seeding."""

    _NAME = "DataflowSystemTerraSeedControlDeltaDataOnly20260920"


class DataflowSystemTerraSeedDeltaDataOnly20260920(_SeedProbeBase):
    """Seeded roots, DELTA, Data-only evidence."""

    _SEED = True
    _NAME = "DataflowSystemTerraSeedDeltaDataOnly20260920"


class DataflowSystemTerraSeedOpBlockDataOnly20260920(_SeedProbeBase):
    """Seeded roots on the retained-observation LATEST layout."""

    _CONTEXT_MODE = "latest"
    _MESSAGE_LAYOUT = "opBlock"
    _SEED = True
    _NAME = "DataflowSystemTerraSeedOpBlockDataOnly20260920"


SEED_PROBE_ARMS = (
    DataflowSystemTerraSeedControlDeltaDataOnly20260920,
    DataflowSystemTerraSeedDeltaDataOnly20260920,
    DataflowSystemTerraSeedOpBlockDataOnly20260920,
)
__all__ = [cls.__name__ for cls in SEED_PROBE_ARMS]

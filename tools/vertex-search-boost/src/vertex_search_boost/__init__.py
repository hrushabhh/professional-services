# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Boost & Bury Python SDK for Google Cloud Vertex AI Search."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vertex_search_boost.models import (
    BOOST_MAX,
    BOOST_MIN,
    DEFAULT_CONFIG_PATH,
    BoostBundleConfig,
    BoostRule,
    ConditionBoostRule,
    ControlPointSpec,
    FreshnessBoostRule,
    NumericalBoostRule,
    SimulationPoint,
    SimulationReport,
    ValidationError,
    WorkspaceTarget,
    find_zero_crossings,
    interpolate_numerical_boost,
    load_boost_bundle,
    parse_bundle_dict,
    parse_rule_dict,
    simulate_numerical_curve,
)

if TYPE_CHECKING:
    from vertex_search_boost.manager import (
        ControlSyncOutcome,
        SchemaFieldInspection,
        ServingConfigAttachment,
        VertexBoostManager,
    )

__version__ = "0.1.0"
_MANAGER_EXPORTS = {"ControlSyncOutcome", "SchemaFieldInspection", "ServingConfigAttachment", "VertexBoostManager"}

__all__ = [
    "BOOST_MAX", "BOOST_MIN", "DEFAULT_CONFIG_PATH", "BoostBundleConfig", "BoostRule", "ConditionBoostRule",
    "ControlPointSpec", "ControlSyncOutcome", "FreshnessBoostRule", "NumericalBoostRule",
    "SchemaFieldInspection", "ServingConfigAttachment", "SimulationPoint", "SimulationReport",
    "ValidationError", "VertexBoostManager", "WorkspaceTarget", "__version__",
    "find_zero_crossings", "interpolate_numerical_boost", "load_boost_bundle",
    "parse_bundle_dict", "parse_rule_dict", "simulate_numerical_curve",
]


def __getattr__(name: str) -> Any:
    if name in _MANAGER_EXPORTS:
        from vertex_search_boost import manager
        return getattr(manager, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

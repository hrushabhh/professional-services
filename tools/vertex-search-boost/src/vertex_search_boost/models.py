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

"""Validated configuration models, JSON loader, and offline curve simulator for Vertex AI Search Boost & Bury."""

from __future__ import annotations

import json
import math
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

BOOST_MIN, BOOST_MAX = -1.0, 1.0
MAX_CONFIG_BYTES = 1_048_576
DEFAULT_CONFIG_PATH = Path(__file__).with_name("multi_signal_boost_bundle.json")

_RESOURCE_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
_LOCATION_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,39}$")
_FIELD_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.]*$")
_DURATION_RE = re.compile(
    r"^P?(?:(\d+(?:\.\d+)?)[dD])?(?:T?(?:(\d+(?:\.\d+)?)[hH])?(?:(\d+(?:\.\d+)?)[mM])?(?:(\d+(?:\.\d+)?)[sS])?)?$"
)


class ValidationError(ValueError):
    """Raised when a boost rule or target resource identifier fails validation."""


def validate_resource_id(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label} must be a non-empty string.")
    cleaned = value.strip()
    if any(sep in cleaned for sep in ("..", "/", "\\", "\x00")) or not _RESOURCE_ID_RE.match(cleaned):
        raise ValidationError(f"Invalid {label}: {value!r}.")
    return cleaned


def validate_location(value: str) -> str:
    if not isinstance(value, str) or not _LOCATION_RE.match(value.strip().lower()):
        raise ValidationError(f"Invalid location {value!r} (expected e.g. 'global', 'us', 'eu').")
    return value.strip().lower()


def validate_field_name(value: str) -> str:
    if not isinstance(value, str) or ".." in value or not _FIELD_NAME_RE.match(value.strip()):
        raise ValidationError(f"Invalid schema field_name: {value!r}.")
    return value.strip()


def validate_boost_amount(value: Union[int, float], label: str = "boost_amount") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValidationError(f"{label} must be a finite float or int.")
    numeric = float(value)
    if not (BOOST_MIN <= numeric <= BOOST_MAX):
        raise ValidationError(f"{label} must be within [{BOOST_MIN}, {BOOST_MAX}], got {numeric}.")
    return 0.0 if abs(numeric) < 1e-12 else numeric


def validate_display_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError("display_name must be a non-empty string.")
    cleaned = value.strip()
    if len(cleaned) > 120 or any(ord(ch) < 32 for ch in cleaned):
        raise ValidationError("display_name must be <= 120 printable characters.")
    return cleaned


def _clean_filter(expr: str, *, required: bool = False) -> str:
    if not isinstance(expr, str) or (required and not expr.strip()):
        raise ValidationError("Filter expression must be a non-empty string.")
    cleaned = expr.strip()
    if len(cleaned) > 2000 or "\x00" in cleaned:
        raise ValidationError("Filter expression exceeds 2000 chars or contains null bytes.")
    return cleaned


def _parse_duration_seconds(raw: str) -> float:
    """Parse simple ('30D', '24H') or ISO-8601 ('2DT12H', 'P7D', 'PT30M') duration to seconds."""
    m = _DURATION_RE.match(raw) if isinstance(raw, str) else None
    if m and any(m.groups()):
        days, hours, mins, secs = (float(g) if g else 0.0 for g in m.groups())
        total = days * 86400.0 + hours * 3600.0 + mins * 60.0 + secs
        if total > 0:
            return total
    raise ValidationError(f"Invalid freshness duration {raw!r} (expected e.g. '7D', '30D', '2DT12H', or '86400s').")


def _ensure_str_sequence(value: Any, label: str) -> Tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError(f"{label} must be a list or tuple of strings.")
    return tuple(value)


@dataclass(frozen=True)
class ControlPointSpec:
    """Single (attribute_value, boost_amount) anchor on a linear boost/bury curve."""

    attribute_value: str
    boost_amount: float

    def __post_init__(self) -> None:
        if isinstance(self.attribute_value, (int, float)) and not isinstance(self.attribute_value, bool):
            if not math.isfinite(float(self.attribute_value)):
                raise ValidationError("attribute_value must be finite.")
            attr = str(self.attribute_value)
        elif isinstance(self.attribute_value, str) and self.attribute_value.strip():
            attr = self.attribute_value.strip()
        else:
            raise ValidationError("attribute_value must be a non-empty string, int, or float.")
        object.__setattr__(self, "attribute_value", attr)
        object.__setattr__(self, "boost_amount", validate_boost_amount(self.boost_amount))

    @property
    def numeric_value(self) -> float:
        try:
            val = float(self.attribute_value)
        except ValueError as exc:
            raise ValidationError(f"Non-numeric attribute_value {self.attribute_value!r}.") from exc
        if not math.isfinite(val):
            raise ValidationError("Numeric attribute_value must be finite.")
        return val


@dataclass(frozen=True)
class WorkspaceTarget:
    """Target Google Cloud Vertex AI Search DataStore and Engine/ServingConfig scope."""

    project_id: str
    location: str
    data_store_id: str
    collection_id: str = "default_collection"
    engine_ids: Tuple[str, ...] = field(default_factory=tuple)
    serving_config_ids: Tuple[str, ...] = ("default_search",)

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_id", validate_resource_id(self.project_id, "project_id"))
        object.__setattr__(self, "location", validate_location(self.location))
        object.__setattr__(self, "data_store_id", validate_resource_id(self.data_store_id, "data_store_id"))
        object.__setattr__(self, "collection_id", validate_resource_id(self.collection_id, "collection_id"))
        eng_seq = _ensure_str_sequence(self.engine_ids, "engine_ids")
        sc_seq = _ensure_str_sequence(self.serving_config_ids, "serving_config_ids")
        object.__setattr__(self, "engine_ids", tuple(dict.fromkeys(validate_resource_id(e, "engine_id") for e in eng_seq)))
        if not sc_seq:
            raise ValidationError("serving_config_ids must not be empty.")
        object.__setattr__(
            self, "serving_config_ids", tuple(dict.fromkeys(validate_resource_id(s, "serving_config_id") for s in sc_seq))
        )

    @property
    def collection_path(self) -> str:
        return f"projects/{self.project_id}/locations/{self.location}/collections/{self.collection_id}"

    @property
    def data_store_path(self) -> str:
        return f"{self.collection_path}/dataStores/{self.data_store_id}"

    @property
    def api_endpoint(self) -> Optional[str]:
        return None if self.location == "global" else f"{self.location}-discoveryengine.googleapis.com"


def _init_interp_rule(rule: Any, label: str, key_fn: Callable[[ControlPointSpec], float]) -> None:
    fname = validate_field_name(rule.field_name)
    object.__setattr__(rule, "control_id", validate_resource_id(rule.control_id, "control_id"))
    object.__setattr__(rule, "field_name", fname)
    object.__setattr__(rule, "display_name", validate_display_name(rule.display_name or f"{fname} {label} Boost"))
    object.__setattr__(rule, "condition", _clean_filter(rule.condition))
    pts = tuple(rule.control_points)
    if len(pts) < 2 or not all(isinstance(pt, ControlPointSpec) for pt in pts):
        raise ValidationError(f"{type(rule).__name__} requires at least 2 ControlPointSpec items.")
    keys = [key_fn(pt) for pt in pts]
    if any(keys[i] <= keys[i - 1] for i in range(1, len(keys))):
        raise ValidationError(f"{type(rule).__name__} control_points must be strictly ascending.")
    object.__setattr__(rule, "control_points", pts)


@dataclass(frozen=True)
class NumericalBoostRule:
    """Piecewise-linear boost/bury rule over an indexable numerical field."""

    control_id: str
    field_name: str
    control_points: Tuple[ControlPointSpec, ...]
    display_name: str = ""
    condition: str = ""

    def __post_init__(self) -> None:
        _init_interp_rule(self, "Numerical", lambda pt: pt.numeric_value)


@dataclass(frozen=True)
class FreshnessBoostRule:
    """Piecewise-linear recency/age decay rule over an indexable datetime field."""

    control_id: str
    field_name: str
    control_points: Tuple[ControlPointSpec, ...]
    display_name: str = ""
    condition: str = ""

    def __post_init__(self) -> None:
        _init_interp_rule(self, "Freshness", lambda pt: _parse_duration_seconds(pt.attribute_value))


@dataclass(frozen=True)
class ConditionBoostRule:
    """Boolean filter condition boost or bury rule with a fixed score adjustment."""

    control_id: str
    filter_expression: str
    boost_amount: float
    display_name: str = ""

    def __post_init__(self) -> None:
        cid = validate_resource_id(self.control_id, "control_id")
        object.__setattr__(self, "control_id", cid)
        object.__setattr__(self, "filter_expression", _clean_filter(self.filter_expression, required=True))
        object.__setattr__(self, "boost_amount", validate_boost_amount(self.boost_amount))
        object.__setattr__(self, "display_name", validate_display_name(self.display_name or f"{cid} Condition Boost"))


BoostRule = Union[NumericalBoostRule, FreshnessBoostRule, ConditionBoostRule]


@dataclass(frozen=True)
class BoostBundleConfig:
    """Declarative bundle combining a target workspace and 1+ boost/bury rules."""

    target: WorkspaceTarget
    rules: Tuple[BoostRule, ...]
    replace_existing_controls: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.target, WorkspaceTarget):
            raise ValidationError("target must be a WorkspaceTarget.")
        if not isinstance(self.replace_existing_controls, bool):
            raise ValidationError("replace_existing_controls must be a boolean.")
        rules_t = tuple(self.rules)
        if not rules_t:
            raise ValidationError("rules must contain at least one rule.")
        seen = set()
        for r in rules_t:
            if not isinstance(r, (NumericalBoostRule, FreshnessBoostRule, ConditionBoostRule)):
                raise ValidationError(f"Unsupported rule type: {type(r).__name__}.")
            if r.control_id in seen:
                raise ValidationError(f"Duplicate control_id: {r.control_id!r}.")
            seen.add(r.control_id)
        object.__setattr__(self, "rules", rules_t)


def _parse_points(raw: Any) -> Tuple[ControlPointSpec, ...]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or len(raw) < 2:
        raise ValidationError("control_points must be a list of at least 2 objects.")
    pts = []
    for i, item in enumerate(raw):
        if not isinstance(item, Mapping) or "attribute_value" not in item or "boost_amount" not in item:
            raise ValidationError(f"control_points[{i}] must contain 'attribute_value' and 'boost_amount'.")
        pts.append(ControlPointSpec(item["attribute_value"], item["boost_amount"]))
    return tuple(pts)


def parse_rule_dict(raw: Mapping[str, Any]) -> BoostRule:
    if not isinstance(raw, Mapping) or not isinstance(raw.get("type"), str):
        raise ValidationError("Rule must be an object with 'type' in ('numerical', 'freshness', 'condition').")
    rtype = raw["type"].strip().lower()
    cid, dname = raw.get("control_id", ""), raw.get("display_name", "") or ""
    if rtype in ("numerical", "freshness"):
        cls = NumericalBoostRule if rtype == "numerical" else FreshnessBoostRule
        return cls(cid, raw.get("field_name", ""), _parse_points(raw.get("control_points")), dname, raw.get("condition", "") or "")
    if rtype in ("condition", "filter"):
        if "boost_amount" not in raw:
            raise ValidationError("Condition rule requires 'boost_amount'.")
        return ConditionBoostRule(cid, raw.get("filter_expression", raw.get("condition", "")), raw["boost_amount"], dname)
    raise ValidationError(f"Unsupported rule type: {raw['type']!r}.")


def parse_bundle_dict(data: Mapping[str, Any]) -> BoostBundleConfig:
    if not isinstance(data, Mapping):
        raise ValidationError("Top-level configuration must be a JSON object.")
    raw_t = data.get("target")
    if not isinstance(raw_t, Mapping):
        raise ValidationError("Configuration must include a 'target' object.")
    raw_rules = data.get("rules")
    if isinstance(raw_rules, (str, bytes)) or not isinstance(raw_rules, Sequence) or not raw_rules:
        raise ValidationError("Configuration must include a non-empty 'rules' list.")
    replace = data.get("replace_existing_controls", False)
    if not isinstance(replace, bool):
        raise ValidationError("replace_existing_controls must be a boolean.")
    raw_eids = raw_t.get("engine_ids")
    raw_scids = raw_t.get("serving_config_ids")
    target = WorkspaceTarget(
        project_id=raw_t.get("project_id", ""),
        location=raw_t.get("location", "global"),
        data_store_id=raw_t.get("data_store_id", ""),
        collection_id=raw_t.get("collection_id", "default_collection"),
        engine_ids=_ensure_str_sequence(raw_eids, "target.engine_ids") if raw_eids is not None else (),
        serving_config_ids=_ensure_str_sequence(raw_scids, "target.serving_config_ids") if raw_scids is not None else ("default_search",),
    )
    return BoostBundleConfig(target=target, rules=tuple(parse_rule_dict(r) for r in raw_rules), replace_existing_controls=replace)


def load_boost_bundle(config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> BoostBundleConfig:
    path_str = str(config_path)
    if not path_str.strip() or "\x00" in path_str:
        raise ValidationError("Invalid configuration file path.")
    resolved = Path(config_path).expanduser().resolve()
    try:
        st = os.stat(resolved)
    except OSError as exc:
        raise ValidationError(f"Cannot access {resolved}: {exc}") from exc
    if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_CONFIG_BYTES:
        raise ValidationError(f"Config {resolved} must be a regular file <= {MAX_CONFIG_BYTES} bytes.")
    try:
        parsed = json.loads(
            resolved.read_text(encoding="utf-8"),
            parse_constant=lambda tok: (_ for _ in ()).throw(ValidationError(f"Invalid JSON constant {tok!r}.")),
        )
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON in {resolved}: {exc}") from exc
    return parse_bundle_dict(parsed)


@dataclass(frozen=True)
class SimulationPoint:
    input_value: Optional[float]
    boost_amount: float
    effect: str
    regime_note: str


@dataclass(frozen=True)
class SimulationReport:
    control_id: str
    field_name: str
    zero_crossings: Tuple[float, ...]
    points: Tuple[SimulationPoint, ...]
    null_value_point: SimulationPoint

    def format_markdown_table(self) -> str:
        zc = (
            f"`{', '.join(f'{x:.4f}' for x in self.zero_crossings)}`"
            if self.zero_crossings
            else "_None within interior segments_"
        )
        lines = [
            f"### Simulation: `{self.control_id}` (field: `{self.field_name}`)",
            "",
            f"- **Neutral (`0.0`) Crossover Value(s):** {zc}",
            "",
            "| Field Value | Computed Boost | Ranking Effect | Interpolation Regime |",
            "| :--- | ---: | :--- | :--- |",
        ]
        for pt in (*self.points, self.null_value_point):
            val = f"{pt.input_value:g}" if pt.input_value is not None else "NULL / Missing"
            lines.append(f"| `{val}` | `{pt.boost_amount:+.4f}` | {pt.effect} | {pt.regime_note} |")
        return "\n".join(lines)


def _effect(b: float) -> str:
    return "BOOST (Promote)" if b > 1e-9 else ("BURY (Demote)" if b < -1e-9 else "NEUTRAL (Unchanged)")


def interpolate_numerical_boost(rule: NumericalBoostRule, value: Optional[float]) -> SimulationPoint:
    """Compute Vertex AI Search piecewise-linear boost/bury for a numeric value."""
    if not isinstance(rule, NumericalBoostRule):
        raise ValidationError("interpolate_numerical_boost requires a NumericalBoostRule.")
    if value is None:
        return SimulationPoint(None, 0.0, "NEUTRAL (Unchanged)", "Missing/null attribute defaults to 0.0")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValidationError(f"Simulation value must be a finite number or None, got {value!r}.")

    x, pts = float(value), rule.control_points
    x_first, x_last = pts[0].numeric_value, pts[-1].numeric_value
    if x <= x_first:
        note = f"Exact lower anchor ({x_first:g})" if x == x_first else f"Clamped below lower anchor ({x_first:g})"
        return SimulationPoint(x, pts[0].boost_amount, _effect(pts[0].boost_amount), note)
    if x >= x_last:
        note = f"Exact upper anchor ({x_last:g})" if x == x_last else f"Clamped above upper anchor ({x_last:g})"
        return SimulationPoint(x, pts[-1].boost_amount, _effect(pts[-1].boost_amount), note)

    for i in range(len(pts) - 1):
        x0, b0 = pts[i].numeric_value, pts[i].boost_amount
        x1, b1 = pts[i + 1].numeric_value, pts[i + 1].boost_amount
        if math.isclose(x, x0, abs_tol=1e-12):
            return SimulationPoint(x, b0, _effect(b0), f"Exact anchor ({x0:g})")
        if x0 < x <= x1:
            if math.isclose(x, x1, abs_tol=1e-12):
                return SimulationPoint(x, b1, _effect(b1), f"Exact anchor ({x1:g})")
            raw_b = b0 + ((x - x0) / (x1 - x0)) * (b1 - b0)
            b = 0.0 if abs(raw_b) < 1e-12 else raw_b
            return SimulationPoint(x, b, _effect(b), f"Linear segment [{x0:g}, {x1:g}]")
    raise AssertionError("Unreachable segment state")


def find_zero_crossings(rule: NumericalBoostRule) -> Tuple[float, ...]:
    """Compute exact attribute values where the curve crosses 0.0 (Boost <-> Bury)."""
    if not isinstance(rule, NumericalBoostRule):
        raise ValidationError("find_zero_crossings requires a NumericalBoostRule.")
    crossings: List[float] = []
    pts = rule.control_points
    for i in range(len(pts) - 1):
        x0, b0 = pts[i].numeric_value, pts[i].boost_amount
        x1, b1 = pts[i + 1].numeric_value, pts[i + 1].boost_amount
        if math.isclose(b0, 0.0, abs_tol=1e-12) and (not crossings or not math.isclose(crossings[-1], x0, abs_tol=1e-9)):
            crossings.append(x0)
        if (b0 > 0.0 > b1) or (b0 < 0.0 < b1):
            crossings.append(x0 - b0 * (x1 - x0) / (b1 - b0))
        if i == len(pts) - 2 and math.isclose(b1, 0.0, abs_tol=1e-12) and (
            not crossings or not math.isclose(crossings[-1], x1, abs_tol=1e-9)
        ):
            crossings.append(x1)
    return tuple(crossings)


def simulate_numerical_curve(
    rule: NumericalBoostRule,
    sample_values: Optional[Iterable[Optional[float]]] = None,
    steps_per_segment: int = 4,
) -> SimulationReport:
    if not isinstance(rule, NumericalBoostRule) or isinstance(steps_per_segment, bool) or not (1 <= steps_per_segment <= 1000):
        raise ValidationError("Invalid rule or steps_per_segment (must be int 1..1000).")
    crossings = find_zero_crossings(rule)
    if sample_values is None:
        pts = rule.control_points
        x_first, x_last = pts[0].numeric_value, pts[-1].numeric_value
        pad = max(1.0, round((x_last - x_first) * 0.1, 4))
        vals = [x_first - pad, x_last, x_last + pad, *(round(z, 6) for z in crossings)]
        for i in range(len(pts) - 1):
            x0, x1 = pts[i].numeric_value, pts[i + 1].numeric_value
            vals.extend(round(x0 + (x1 - x0) * (s / steps_per_segment), 6) for s in range(steps_per_segment))
        samples: List[float] = sorted(set(vals))
    else:
        samples = [v for v in sample_values if v is not None]
    return SimulationReport(
        control_id=rule.control_id,
        field_name=rule.field_name,
        zero_crossings=crossings,
        points=tuple(interpolate_numerical_boost(rule, v) for v in samples),
        null_value_point=interpolate_numerical_boost(rule, None),
    )

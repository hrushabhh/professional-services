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

"""Discovery Engine manager for Boost & Bury Control lifecycle and inline BoostSpec."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google.api_core import exceptions as gcp_exceptions
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine
from google.protobuf import field_mask_pb2

from vertex_search_boost.models import (
    DEFAULT_CONFIG_PATH,
    BoostBundleConfig,
    BoostRule,
    ConditionBoostRule,
    FreshnessBoostRule,
    NumericalBoostRule,
    ValidationError,
    WorkspaceTarget,
    load_boost_bundle,
    parse_bundle_dict,
    validate_field_name,
    validate_resource_id,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServingConfigAttachment:
    serving_config_name: str
    active_boost_control_ids: Tuple[str, ...]
    status: str  # "updated", "unchanged", "not_found"


@dataclass(frozen=True)
class ControlSyncOutcome:
    engine_name: str
    control_name: str
    control_id: str
    action: str  # "created", "updated"
    attachments: Tuple[ServingConfigAttachment, ...]


@dataclass(frozen=True)
class SchemaFieldInspection:
    data_store_path: str
    field_name: str
    exists_in_schema: bool
    field_type: Optional[str]
    indexable: bool
    dynamic_facetable: bool
    retrievable: bool
    warnings: Tuple[str, ...]

    @property
    def is_ready_for_boost(self) -> bool:
        return self.exists_in_schema and self.indexable


def _build_interp_spec(spec_cls: Any, rule: BoostRule) -> Any:
    attr_type = spec_cls.AttributeType.NUMERICAL if isinstance(rule, NumericalBoostRule) else spec_cls.AttributeType.FRESHNESS
    return spec_cls(
        field_name=rule.field_name,
        attribute_type=attr_type,
        interpolation_type=spec_cls.InterpolationType.LINEAR,
        control_points=[spec_cls.ControlPoint(attribute_value=p.attribute_value, boost_amount=p.boost_amount) for p in rule.control_points],
    )


class VertexBoostManager:
    """SDK manager for Vertex AI Search (Discovery Engine) Boost & Bury controls."""

    def __init__(
        self,
        target: WorkspaceTarget,
        *,
        control_client: Optional[discoveryengine.ControlServiceClient] = None,
        engine_client: Optional[discoveryengine.EngineServiceClient] = None,
        serving_config_client: Optional[discoveryengine.ServingConfigServiceClient] = None,
        schema_client: Optional[discoveryengine.SchemaServiceClient] = None,
    ) -> None:
        if not isinstance(target, WorkspaceTarget):
            raise ValidationError("target must be a WorkspaceTarget instance.")
        self.target = target
        self._opts = ClientOptions(api_endpoint=target.api_endpoint) if target.api_endpoint else None
        self._clients: Dict[str, Any] = {
            "control": control_client,
            "engine": engine_client,
            "sc": serving_config_client,
            "schema": schema_client,
        }

    def _get_client(self, key: str, cls: Any) -> Any:
        if self._clients[key] is None:
            self._clients[key] = cls(client_options=self._opts)
        return self._clients[key]

    @property
    def control_client(self) -> discoveryengine.ControlServiceClient:
        return self._get_client("control", discoveryengine.ControlServiceClient)

    @property
    def engine_client(self) -> discoveryengine.EngineServiceClient:
        return self._get_client("engine", discoveryengine.EngineServiceClient)

    @property
    def serving_config_client(self) -> discoveryengine.ServingConfigServiceClient:
        return self._get_client("sc", discoveryengine.ServingConfigServiceClient)

    @property
    def schema_client(self) -> discoveryengine.SchemaServiceClient:
        return self._get_client("schema", discoveryengine.SchemaServiceClient)

    def resolve_engine_paths(self) -> Tuple[str, ...]:
        if self.target.engine_ids:
            return tuple(f"{self.target.collection_path}/engines/{eid}" for eid in self.target.engine_ids)
        ds_id = self.target.data_store_id
        return tuple(
            eng.name
            for eng in self.engine_client.list_engines(parent=self.target.collection_path)
            if ds_id in eng.data_store_ids or eng.name.rsplit("/", 1)[-1] == ds_id
        )

    def build_control_proto(self, rule: BoostRule) -> discoveryengine.Control:
        if isinstance(rule, (NumericalBoostRule, FreshnessBoostRule)):
            action = discoveryengine.Control.BoostAction(
                data_store=self.target.data_store_path,
                filter=rule.condition,
                interpolation_boost_spec=_build_interp_spec(discoveryengine.Control.BoostAction.InterpolationBoostSpec, rule),
            )
        elif isinstance(rule, ConditionBoostRule):
            boost_key = "fixed_boost" if "fixed_boost" in discoveryengine.Control.BoostAction.meta.fields else "boost"
            action = discoveryengine.Control.BoostAction(
                data_store=self.target.data_store_path, filter=rule.filter_expression, **{boost_key: rule.boost_amount}
            )
        else:
            raise ValidationError(f"Unsupported BoostRule type: {type(rule).__name__}.")

        return discoveryengine.Control(
            display_name=rule.display_name,
            solution_type=discoveryengine.SolutionType.SOLUTION_TYPE_SEARCH,
            use_cases=[discoveryengine.SearchUseCase.SEARCH_USE_CASE_SEARCH],
            boost_action=action,
        )

    @staticmethod
    def build_inline_boost_spec(rules: Sequence[BoostRule]) -> discoveryengine.SearchRequest.BoostSpec:
        if not rules:
            raise ValidationError("rules must contain at least one BoostRule.")
        CondSpec = discoveryengine.SearchRequest.BoostSpec.ConditionBoostSpec
        specs: List[Any] = []
        for rule in rules:
            if isinstance(rule, (NumericalBoostRule, FreshnessBoostRule)):
                specs.append(
                    CondSpec(condition=rule.condition, boost_control_spec=_build_interp_spec(CondSpec.BoostControlSpec, rule))
                )
            elif isinstance(rule, ConditionBoostRule):
                specs.append(CondSpec(condition=rule.filter_expression, boost=rule.boost_amount))
            else:
                raise ValidationError(f"Unsupported BoostRule type: {type(rule).__name__}.")
        return discoveryengine.SearchRequest.BoostSpec(condition_boost_specs=specs)

    def _update_existing_control(self, name: str, proto: discoveryengine.Control) -> Tuple[str, str]:
        proto.name = name
        res = self.control_client.update_control(
            request=discoveryengine.UpdateControlRequest(
                control=proto, update_mask=field_mask_pb2.FieldMask(paths=["display_name", "boost_action"])
            )
        )
        return res.name or name, "updated"

    def _upsert_control(self, engine_path: str, rule: BoostRule) -> Tuple[str, str]:
        proto = self.build_control_proto(rule)
        name = f"{engine_path}/controls/{rule.control_id}"
        try:
            self.control_client.get_control(name=name)
            return self._update_existing_control(name, proto)
        except gcp_exceptions.NotFound:
            try:
                res = self.control_client.create_control(
                    request=discoveryengine.CreateControlRequest(parent=engine_path, control=proto, control_id=rule.control_id)
                )
                return res.name or name, "created"
            except gcp_exceptions.AlreadyExists:
                return self._update_existing_control(name, proto)

    def _patch_serving_config(self, sc_name: str, ids: Tuple[str, ...]) -> ServingConfigAttachment:
        try:
            self.serving_config_client.update_serving_config(
                request=discoveryengine.UpdateServingConfigRequest(
                    serving_config=discoveryengine.ServingConfig(name=sc_name, boost_control_ids=list(ids)),
                    update_mask=field_mask_pb2.FieldMask(paths=["boost_control_ids"]),
                )
            )
            return ServingConfigAttachment(sc_name, ids, "updated")
        except gcp_exceptions.NotFound:
            return ServingConfigAttachment(sc_name, (), "not_found")

    def _attach_controls(
        self, engine_path: str, sc_id: str, new_ids: Sequence[str], *, replace_existing: bool = False
    ) -> ServingConfigAttachment:
        sc_name = f"{engine_path}/servingConfigs/{sc_id}"
        existing: List[str] = []
        if not replace_existing:
            try:
                existing = list(self.serving_config_client.get_serving_config(name=sc_name).boost_control_ids)
            except gcp_exceptions.NotFound:
                return ServingConfigAttachment(sc_name, (), "not_found")

        merged = tuple(dict.fromkeys(new_ids if replace_existing else [*existing, *new_ids]))
        if not replace_existing and tuple(existing) == merged:
            return ServingConfigAttachment(sc_name, merged, "unchanged")
        return self._patch_serving_config(sc_name, merged)

    def sync_rule(self, rule: BoostRule, *, replace_existing_controls: bool = False) -> Tuple[ControlSyncOutcome, ...]:
        return self.sync_rules([rule], replace_existing_controls=replace_existing_controls)

    def sync_rules(self, rules: Sequence[BoostRule], *, replace_existing_controls: bool = False) -> Tuple[ControlSyncOutcome, ...]:
        if not rules:
            raise ValidationError("rules must contain at least one BoostRule.")
        engine_paths = self.resolve_engine_paths()
        if not engine_paths:
            logger.warning("No Search Engines found linked to DataStore '%s'.", self.target.data_store_id)
            return ()

        outcomes: List[ControlSyncOutcome] = []
        new_ids = [r.control_id for r in rules]
        for engine_path in engine_paths:
            upserted = [(r, *self._upsert_control(engine_path, r)) for r in rules]
            attachments = tuple(
                self._attach_controls(engine_path, sc_id, new_ids, replace_existing=replace_existing_controls)
                for sc_id in self.target.serving_config_ids
            )
            outcomes.extend(
                ControlSyncOutcome(engine_path, ctrl_name, r.control_id, action, attachments)
                for r, ctrl_name, action in upserted
            )
        return tuple(outcomes)

    @classmethod
    def sync_bundle(cls, bundle: Optional[Any] = None, **client_overrides: Any) -> Tuple[ControlSyncOutcome, ...]:
        if bundle is None or isinstance(bundle, (str, Path)):
            resolved = load_boost_bundle(DEFAULT_CONFIG_PATH if bundle is None else bundle)
        elif isinstance(bundle, Mapping):
            resolved = parse_bundle_dict(bundle)
        elif isinstance(bundle, BoostBundleConfig):
            resolved = bundle
        else:
            raise ValidationError("bundle must be a BoostBundleConfig, dict, file path, or None.")
        return cls(resolved.target, **client_overrides).sync_rules(
            resolved.rules, replace_existing_controls=resolved.replace_existing_controls
        )

    def detach_control(self, control_id: str, *, delete_resource: bool = False) -> Tuple[ServingConfigAttachment, ...]:
        cid = validate_resource_id(control_id, "control_id")
        results: List[ServingConfigAttachment] = []
        for engine_path in self.resolve_engine_paths():
            for sc_id in self.target.serving_config_ids:
                sc_name = f"{engine_path}/servingConfigs/{sc_id}"
                try:
                    existing = list(self.serving_config_client.get_serving_config(name=sc_name).boost_control_ids)
                except gcp_exceptions.NotFound:
                    results.append(ServingConfigAttachment(sc_name, (), "not_found"))
                    continue
                remaining = tuple(x for x in existing if x != cid)
                if len(remaining) == len(existing):
                    results.append(ServingConfigAttachment(sc_name, remaining, "unchanged"))
                else:
                    results.append(self._patch_serving_config(sc_name, remaining))
            if delete_resource:
                try:
                    self.control_client.delete_control(name=f"{engine_path}/controls/{cid}")
                except gcp_exceptions.NotFound:
                    pass
        return tuple(results)

    def inspect_schema_field(self, field_name: str, *, schema_id: str = "default_schema") -> SchemaFieldInspection:
        fname = validate_field_name(field_name)
        schema_path = f"{self.target.data_store_path}/schemas/{validate_resource_id(schema_id, 'schema_id')}"
        resp = self.schema_client.get_schema(name=schema_path)
        raw_json = getattr(resp, "json_schema", "") or ""
        if raw_json:
            try:
                node: Any = json.loads(raw_json)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"Invalid json_schema in {schema_path}: {exc}") from exc
        else:
            struct_schema = getattr(resp, "struct_schema", None)
            node = dict(struct_schema) if isinstance(struct_schema, Mapping) else {}

        for seg in fname.split("."):
            if not isinstance(node, Mapping):
                node = None
                break
            props = node.get("properties")
            if not isinstance(props, Mapping) and isinstance(node.get("items"), Mapping):
                props = node["items"].get("properties")
            node = props.get(seg) if isinstance(props, Mapping) else None

        if not isinstance(node, Mapping):
            return SchemaFieldInspection(
                self.target.data_store_path, fname, False, None, False, False, False,
                (f"Field '{fname}' not found in {schema_path}; unindexed fields evaluate as 0.0 (neutral).",),
            )

        has_key_mapping = bool(node.get("keyPropertyMapping"))
        indexable = bool(node.get("indexable", has_key_mapping))
        dyn_facet = bool(node.get("dynamicFacetable", False))
        warnings: List[str] = []
        if not indexable:
            warnings.append(f"Field '{fname}' has indexable=False; Boost/Bury requires indexable=True.")
        if dyn_facet:
            warnings.append(f"Field '{fname}' has dynamicFacetable=True; disable on pure ranking fields to avoid facet payload bloat.")
        return SchemaFieldInspection(
            self.target.data_store_path, fname, True,
            str(node["type"]) if node.get("type") is not None else None,
            indexable, dyn_facet, bool(node.get("retrievable", False)), tuple(warnings),
        )


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    for outcome in VertexBoostManager.sync_bundle(cfg_path):
        print(f"[{outcome.action.upper()}] {outcome.control_name}")
        for att in outcome.attachments:
            print(f"  -> {att.serving_config_name} ({att.status}): {list(att.active_boost_control_ids)}")

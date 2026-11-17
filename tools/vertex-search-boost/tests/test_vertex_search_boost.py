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

"""Unit tests for the vertex-search-boost Python SDK."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

# Ensure local `src/` is importable when running via `python3 -m unittest` without installation.
_PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from google.api_core import exceptions as gcp_exceptions
from google.cloud import discoveryengine_v1 as discoveryengine

from vertex_search_boost import (
    BoostBundleConfig,
    ConditionBoostRule,
    ControlPointSpec,
    FreshnessBoostRule,
    NumericalBoostRule,
    ValidationError,
    VertexBoostManager,
    WorkspaceTarget,
    find_zero_crossings,
    interpolate_numerical_boost,
    load_boost_bundle,
    simulate_numerical_curve,
)


class TestModelsAndValidation(unittest.TestCase):
    def test_valid_workspace_target_endpoints(self) -> None:
        global_target = WorkspaceTarget(
            project_id="demo-proj-01",
            location="global",
            data_store_id="docs-store",
        )
        self.assertIsNone(global_target.api_endpoint)
        self.assertEqual(
            global_target.data_store_path,
            "projects/demo-proj-01/locations/global/collections/default_collection/dataStores/docs-store",
        )

        regional_target = WorkspaceTarget(
            project_id="demo-proj-01",
            location="us",
            data_store_id="docs-store",
        )
        self.assertEqual(
            regional_target.api_endpoint, "us-discoveryengine.googleapis.com"
        )

    def test_rejects_path_traversal_and_invalid_ids(self) -> None:
        with self.assertRaises(ValidationError):
            WorkspaceTarget(
                project_id="../secret-project",
                location="global",
                data_store_id="docs-store",
            )
        with self.assertRaises(ValidationError):
            WorkspaceTarget(
                project_id="demo-proj",
                location="global/../../evil",
                data_store_id="docs-store",
            )
        with self.assertRaises(ValidationError):
            NumericalBoostRule(
                control_id="ctrl/evil",
                field_name="rank_score",
                control_points=(ControlPointSpec("1", 0.5), ControlPointSpec("5", 0.0)),
            )
        with self.assertRaises(ValidationError):
            NumericalBoostRule(
                control_id="valid-ctrl",
                field_name="invalid-field-with-hyphen",
                control_points=(ControlPointSpec("1", 0.5), ControlPointSpec("5", 0.0)),
            )

    def test_rejects_out_of_bounds_or_non_ascending_points(self) -> None:
        with self.assertRaises(ValidationError):
            ControlPointSpec("1", 1.25)
        with self.assertRaises(ValidationError):
            ControlPointSpec("1", -1.01)
        with self.assertRaises(ValidationError):
            ControlPointSpec("1", float("nan"))
        with self.assertRaises(ValidationError):
            ControlPointSpec("1", True)  # type: ignore[arg-type]

        with self.assertRaises(ValidationError):
            NumericalBoostRule(
                control_id="bad-order",
                field_name="tier_rank",
                control_points=(
                    ControlPointSpec("10", 0.5),
                    ControlPointSpec("2", -0.2),
                ),
            )

    def test_freshness_rule_orders_durations_correctly(self) -> None:
        rule = FreshnessBoostRule(
            control_id="recency-rule",
            field_name="updated_at",
            control_points=(
                ControlPointSpec("24H", 0.6),
                ControlPointSpec("30D", 0.2),
                ControlPointSpec("365D", -0.2),
            ),
        )
        self.assertEqual(len(rule.control_points), 3)

        with self.assertRaises(ValidationError):
            FreshnessBoostRule(
                control_id="bad-recency",
                field_name="updated_at",
                control_points=(
                    ControlPointSpec("365D", 0.5),
                    ControlPointSpec("30D", 0.0),
                ),
            )


class TestSimulatorMath(unittest.TestCase):
    def test_linear_interpolation_clamping_null_and_zero_crossing(self) -> None:
        rule = NumericalBoostRule(
            control_id="tier-curve",
            field_name="quality_tier",
            control_points=(
                ControlPointSpec("1", 0.8),
                ControlPointSpec("9", -0.8),
            ),
        )

        # Clamped below lower bound
        self.assertAlmostEqual(interpolate_numerical_boost(rule, -5).boost_amount, 0.8)
        # Exact lower bound
        self.assertAlmostEqual(interpolate_numerical_boost(rule, 1).boost_amount, 0.8)
        # Midpoint x=5 should cross 0.0 exactly
        mid = interpolate_numerical_boost(rule, 5)
        self.assertAlmostEqual(mid.boost_amount, 0.0)
        self.assertEqual(mid.effect, "NEUTRAL (Unchanged)")
        # Interior bury x=7 -> -0.4
        self.assertAlmostEqual(interpolate_numerical_boost(rule, 7).boost_amount, -0.4)
        # Clamped above upper bound
        self.assertAlmostEqual(interpolate_numerical_boost(rule, 50).boost_amount, -0.8)
        # Missing / None attribute -> 0.0 neutral
        null_pt = interpolate_numerical_boost(rule, None)
        self.assertEqual(null_pt.boost_amount, 0.0)
        self.assertEqual(null_pt.effect, "NEUTRAL (Unchanged)")

        crossings = find_zero_crossings(rule)
        self.assertEqual(len(crossings), 1)
        self.assertAlmostEqual(crossings[0], 5.0)

        report = simulate_numerical_curve(rule, sample_values=[0, 1, 5, 9, 12])
        md = report.format_markdown_table()
        self.assertIn("`quality_tier`", md)
        self.assertIn("`NULL / Missing`", md)


class TestVertexBoostManager(unittest.TestCase):
    def setUp(self) -> None:
        self.target = WorkspaceTarget(
            project_id="acme-search-prod",
            location="global",
            data_store_id="kb-datastore",
            serving_config_ids=("default_search",),
        )
        self.mock_control = MagicMock()
        self.mock_engine = MagicMock()
        self.mock_sc = MagicMock()
        self.mock_schema = MagicMock()

        self.engine_full_name = (
            "projects/acme-search-prod/locations/global"
            "/collections/default_collection/engines/kb-engine"
        )
        self.mock_engine.list_engines.return_value = [
            SimpleNamespace(
                name=self.engine_full_name,
                data_store_ids=["kb-datastore"],
            )
        ]

        self.manager = VertexBoostManager(
            self.target,
            control_client=self.mock_control,
            engine_client=self.mock_engine,
            serving_config_client=self.mock_sc,
            schema_client=self.mock_schema,
        )

    def test_sync_creates_new_control_and_preserves_existing_serving_controls(self) -> None:
        rule = NumericalBoostRule(
            control_id="boost-tier",
            field_name="tier_score",
            control_points=(
                ControlPointSpec("1", 0.75),
                ControlPointSpec("10", -0.25),
            ),
        )
        self.mock_control.get_control.side_effect = gcp_exceptions.NotFound("missing")
        self.mock_control.create_control.return_value = SimpleNamespace(
            name=f"{self.engine_full_name}/controls/boost-tier"
        )
        # Existing ServingConfig already has an active control "existing-pin-ctrl"
        self.mock_sc.get_serving_config.return_value = SimpleNamespace(
            boost_control_ids=["existing-pin-ctrl"]
        )

        outcomes = self.manager.sync_rule(rule)
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0].action, "created")
        self.mock_control.create_control.assert_called_once()
        self.mock_control.update_control.assert_not_called()

        # Verify existing control was preserved alongside the new control!
        self.mock_sc.update_serving_config.assert_called_once()
        update_req = self.mock_sc.update_serving_config.call_args.kwargs["request"]
        self.assertEqual(
            list(update_req.serving_config.boost_control_ids),
            ["existing-pin-ctrl", "boost-tier"],
        )

    def test_sync_updates_existing_control_with_field_mask(self) -> None:
        rule = ConditionBoostRule(
            control_id="bury-stale",
            filter_expression='status: ANY("archived")',
            boost_amount=-0.5,
        )
        self.mock_control.get_control.return_value = SimpleNamespace(
            name=f"{self.engine_full_name}/controls/bury-stale"
        )
        self.mock_control.update_control.return_value = SimpleNamespace(
            name=f"{self.engine_full_name}/controls/bury-stale"
        )
        self.mock_sc.get_serving_config.return_value = SimpleNamespace(
            boost_control_ids=["bury-stale"]
        )

        outcomes = self.manager.sync_rule(rule)
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0].action, "updated")
        self.mock_control.update_control.assert_called_once()
        self.mock_control.create_control.assert_not_called()
        # Since "bury-stale" is already attached, ServingConfig update is skipped ("unchanged")
        self.assertEqual(outcomes[0].attachments[0].status, "unchanged")
        self.mock_sc.update_serving_config.assert_not_called()

    def test_detach_control_and_optional_delete(self) -> None:
        self.mock_sc.get_serving_config.return_value = SimpleNamespace(
            boost_control_ids=["keep-ctrl", "remove-ctrl"]
        )
        results = self.manager.detach_control("remove-ctrl", delete_resource=True)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "updated")
        self.assertEqual(results[0].active_boost_control_ids, ("keep-ctrl",))
        self.mock_control.delete_control.assert_called_once_with(
            name=f"{self.engine_full_name}/controls/remove-ctrl"
        )

    def test_build_inline_boost_spec_for_all_three_rule_types(self) -> None:
        rules = [
            NumericalBoostRule(
                control_id="num-ctrl",
                field_name="rank_num",
                control_points=(ControlPointSpec("1", 0.9), ControlPointSpec("5", -0.2)),
            ),
            FreshnessBoostRule(
                control_id="fresh-ctrl",
                field_name="published_time",
                control_points=(ControlPointSpec("7D", 0.5), ControlPointSpec("180D", -0.1)),
            ),
            ConditionBoostRule(
                control_id="cond-ctrl",
                filter_expression='category: ANY("featured")',
                boost_amount=0.4,
            ),
        ]
        spec = VertexBoostManager.build_inline_boost_spec(rules)
        self.assertIsInstance(spec, discoveryengine.SearchRequest.BoostSpec)
        self.assertEqual(len(spec.condition_boost_specs), 3)
        self.assertEqual(
            spec.condition_boost_specs[0].boost_control_spec.field_name, "rank_num"
        )
        self.assertEqual(
            spec.condition_boost_specs[1].boost_control_spec.field_name, "published_time"
        )
        self.assertAlmostEqual(spec.condition_boost_specs[2].boost, 0.4, places=5)

    def test_inspect_schema_field_flags_unindexed_and_dynamic_facetable(self) -> None:
        schema_json = json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "authority_tier": {
                        "type": "number",
                        "indexable": True,
                        "dynamicFacetable": True,
                        "retrievable": False,
                    },
                    "unindexed_score": {
                        "type": "number",
                        "indexable": False,
                    },
                },
            }
        )
        self.mock_schema.get_schema.return_value = SimpleNamespace(json_schema=schema_json)

        ready_info = self.manager.inspect_schema_field("authority_tier")
        self.assertTrue(ready_info.is_ready_for_boost)
        self.assertTrue(ready_info.dynamic_facetable)
        self.assertEqual(len(ready_info.warnings), 1)
        self.assertIn("dynamicFacetable=True", ready_info.warnings[0])

        unindexed_info = self.manager.inspect_schema_field("unindexed_score")
        self.assertFalse(unindexed_info.is_ready_for_boost)
        self.assertIn("indexable=False", unindexed_info.warnings[0])

    def test_inspect_schema_supports_struct_schema_and_key_property_mapping(self) -> None:
        struct_schema = {
            "type": "object",
            "properties": {
                "doc_title": {
                    "type": "string",
                    "keyPropertyMapping": "title",
                }
            },
        }
        self.mock_schema.get_schema.return_value = SimpleNamespace(
            json_schema="", struct_schema=struct_schema
        )
        info = self.manager.inspect_schema_field("doc_title")
        self.assertTrue(info.exists_in_schema)
        self.assertTrue(info.indexable)
        self.assertTrue(info.is_ready_for_boost)

    def test_upsert_control_recovers_from_already_exists_race(self) -> None:
        rule = ConditionBoostRule(
            control_id="ctrl-race",
            filter_expression='status: ANY("published")',
            boost_amount=0.4,
        )
        self.mock_control.get_control.side_effect = gcp_exceptions.NotFound("not found")
        self.mock_control.create_control.side_effect = gcp_exceptions.AlreadyExists("already exists")
        self.mock_sc.get_serving_config.return_value = SimpleNamespace(boost_control_ids=[])

        results = self.manager.sync_rules([rule])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].action, "updated")
        self.mock_control.update_control.assert_called_once()


class TestCodeReviewImprovements(unittest.TestCase):
    def test_compound_iso8601_duration_parsing_and_zero_rejection(self) -> None:
        rule = FreshnessBoostRule(
            control_id="fresh-compound",
            field_name="updated_at",
            control_points=(
                ControlPointSpec("2D", 0.6),
                ControlPointSpec("2DT12H", 0.3),
                ControlPointSpec("P7D", 0.0),
            ),
        )
        self.assertEqual(len(rule.control_points), 3)
        with self.assertRaises(ValidationError):
            FreshnessBoostRule(
                control_id="fresh-zero",
                field_name="updated_at",
                control_points=(ControlPointSpec("0D", 0.5),),
            )

    def test_workspace_target_rejects_bare_string_sequences(self) -> None:
        with self.assertRaises(ValidationError):
            WorkspaceTarget(
                project_id="my-gcp-project-123",
                location="global",
                data_store_id="ds-main",
                serving_config_ids="default_search",  # type: ignore[arg-type]
            )

    def test_simulate_numerical_curve_deduplicates_none_in_sample_values(self) -> None:
        rule = NumericalBoostRule(
            control_id="num-dedup",
            field_name="tier",
            control_points=(ControlPointSpec(1, 0.5), ControlPointSpec(5, -0.5)),
        )
        report = simulate_numerical_curve(rule, sample_values=[1, None, 5])
        md = report.format_markdown_table()
        self.assertEqual(md.count("NULL / Missing"), 1)


class TestConfigLoader(unittest.TestCase):
    def test_load_example_bundle_simulate_and_compile_protos(self) -> None:
        bundle = load_boost_bundle()
        self.assertIsInstance(bundle, BoostBundleConfig)
        self.assertEqual(len(bundle.rules), 3)

        num_rules = [r for r in bundle.rules if isinstance(r, NumericalBoostRule)]
        self.assertEqual(len(num_rules), 1)
        report = simulate_numerical_curve(num_rules[0])
        self.assertIn("authority_tier", report.format_markdown_table())

        mgr = VertexBoostManager(bundle.target)
        controls = [mgr.build_control_proto(r) for r in bundle.rules]
        self.assertEqual(len(controls), 3)
        inline_spec = mgr.build_inline_boost_spec(bundle.rules)
        self.assertEqual(len(inline_spec.condition_boost_specs), 3)

    def test_config_loader_rejects_oversized_or_nan_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            nan_file = os.path.join(tmpdir, "bad_nan.json")
            with open(nan_file, "w", encoding="utf-8") as fh:
                fh.write('{"type": "condition", "control_id": "c1", "filter_expression": "a: ANY(\\"x\\")", "boost_amount": NaN}')
            with self.assertRaises(ValidationError):
                load_boost_bundle(nan_file)


if __name__ == "__main__":
    unittest.main()


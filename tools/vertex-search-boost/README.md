# Vertex AI Search Boost (`vertex-search-boost`)

A lightweight Python SDK for configuring, simulating, and syncing **Boost & Bury** ranking controls in **Google Cloud Vertex AI Search (Discovery Engine)**.

Out of the box, Vertex AI Search ranks documents by semantic and keyword similarity—which often lets outdated `v1` pages, drafts, or low-authority notes outrank canonical documentation. `vertex-search-boost` lets you shape search and RAG rankings at query time (**with zero re-indexing**) using validated, idempotent Python rules.

---

## Key Features

| Capability | Description |
| :--- | :--- |
| **All 3 Boost & Bury Modes** | Supports numerical interpolation curves (`NumericalBoostRule`), recency/age decay curves (`FreshnessBoostRule`), and boolean filter promotions/penalties (`ConditionBoostRule`). |
| **Offline Curve Simulator** | Previews exact piecewise-linear slopes, clamping boundaries, `NULL` (`0.0`) fallbacks, and Boost-to-Bury (`0.0`) crossover points locally without GCP credentials. |
| **Idempotent & Non-Destructive Sync** | Creates or updates Discovery Engine `Control` resources via `FieldMask` and merges `boost_control_ids` onto `ServingConfig` without overwriting existing controls. |
| **Engine + Direct RAG Parity** | Compiles the exact same rules into persistent Engine `Control` resources (`sync_rules`) **and** inline `SearchRequest.BoostSpec` payloads (`build_inline_boost_spec`) for direct `DataStore` RAG queries. |
| **Schema Readiness Checks** | Verifies that target fields have `indexable: true` in `default_schema` and warns when `dynamicFacetable: true` adds unnecessary query overhead. |

---

## Prerequisites & Installation

* **Python:** `>= 3.9`
* **Google Cloud Authentication:** Application Default Credentials (`gcloud auth application-default login`) and IAM role `roles/discoveryengine.admin` (only required when syncing controls to GCP; offline simulation requires no credentials).

```bash
# Install package in editable mode (with test dependencies)
pip install -e ".[dev]"
```

---

## Quickstart

### 1. Define Rules & Simulate Curves Offline

Use `simulate_numerical_curve()` to verify exact interpolated boost scores and zero-crossings for `NumericalBoostRule` before deploying to Google Cloud:

```python
from vertex_search_boost import (
    ConditionBoostRule,
    ControlPointSpec,
    FreshnessBoostRule,
    NumericalBoostRule,
    simulate_numerical_curve,
)

# 1. Numerical curve (authority tier 1 -> +0.85, 5 -> 0.0, 10+ -> -0.35)
authority_rule = NumericalBoostRule(
    control_id="boost-authority-tier",
    field_name="authority_tier",
    control_points=(
        ControlPointSpec("1", 0.85),
        ControlPointSpec("5", 0.00),
        ControlPointSpec("10", -0.35),
    ),
)

# 2. Freshness decay curve (durations on an indexable datetime/timestamp field)
recency_rule = FreshnessBoostRule(
    control_id="boost-recency-decay",
    field_name="published_at",
    control_points=(
        ControlPointSpec("30D", 0.45),
        ControlPointSpec("365D", 0.00),
        ControlPointSpec("1825D", -0.25),
    ),
)

# 3. Filter condition boost/bury
deprecated_rule = ConditionBoostRule(
    control_id="bury-deprecated-docs",
    filter_expression='lifecycle_status: ANY("deprecated", "superseded")',
    boost_amount=-0.60,
)

# Preview interpolated points & zero-crossings offline
report = simulate_numerical_curve(
    authority_rule,
    sample_values=[0, 1, 3, 5, 7.5, 10, 15],
)
print(report.format_markdown_table())
print("Boost-to-Bury crossover at:", report.zero_crossings)  # (5.0,)
```

**Simulated Output:**

| Field Value | Computed Boost | Ranking Effect | Interpolation Regime |
| :--- | ---: | :--- | :--- |
| `0` | `+0.8500` | BOOST (Promote) | Clamped below lower anchor (1) |
| `1` | `+0.8500` | BOOST (Promote) | Exact lower anchor (1) |
| `3` | `+0.4250` | BOOST (Promote) | Linear segment [1, 5] |
| `5` | `+0.0000` | NEUTRAL (Unchanged) | Exact anchor (5) |
| `7.5` | `-0.1750` | BURY (Demote) | Linear segment [5, 10] |
| `10` | `-0.3500` | BURY (Demote) | Exact upper anchor (10) |
| `15` | `-0.3500` | BURY (Demote) | Clamped above upper anchor (10) |
| `NULL / Missing` | `+0.0000` | NEUTRAL (Unchanged) | Missing/null attribute defaults to 0.0 |

---

### 2. Inspect Schema & Sync Controls to Vertex AI Search

Point `WorkspaceTarget` at your GCP project and `DataStore`. `VertexBoostManager` auto-discovers attached Search Engines, creates or updates each `Control`, and merges them onto `default_search` without wiping out other active controls:

```python
from vertex_search_boost import VertexBoostManager, WorkspaceTarget

target = WorkspaceTarget(
    project_id="my-gcp-project",
    location="global",                       # or regional: "us", "eu"
    data_store_id="enterprise-kb-datastore",
    serving_config_ids=("default_search",),  # engine_ids auto-discovered if omitted
)

manager = VertexBoostManager(target)

# Optional pre-flight check: verify field is indexable in default_schema
schema_info = manager.inspect_schema_field("authority_tier")
if not schema_info.is_ready_for_boost:
    raise RuntimeError(schema_info.warnings)

# Create/update controls & attach to default_search
rules = [authority_rule, recency_rule, deprecated_rule]
outcomes = manager.sync_rules(rules)

for outcome in outcomes:
    print(outcome.action, outcome.control_name)
```

---

### 3. Apply Inline `BoostSpec` in Direct DataStore RAG Queries

When your Python RAG service queries `dataStores/.../servingConfigs/default_search:search` directly (instead of an Engine `servingConfig`), pass `VertexBoostManager.build_inline_boost_spec(rules)` on the `SearchRequest`:

```python
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine
from vertex_search_boost import VertexBoostManager

# For regional locations ("us", "eu"), pass target.api_endpoint; for "global", it is None.
client_options = (
    ClientOptions(api_endpoint=target.api_endpoint) if target.api_endpoint else None
)
search_client = discoveryengine.SearchServiceClient(client_options=client_options)

request = discoveryengine.SearchRequest(
    serving_config=f"{target.data_store_path}/servingConfigs/default_search",
    query="production incident escalation policy",
    page_size=10,
    boost_spec=VertexBoostManager.build_inline_boost_spec(rules),
)

response = search_client.search(request=request)
```

---

### 4. Sync from Declarative JSON (`multi_signal_boost_bundle.json`) & Roll Back

Edit [`src/vertex_search_boost/multi_signal_boost_bundle.json`](src/vertex_search_boost/multi_signal_boost_bundle.json) with your GCP `project_id`, `data_store_id`, and schema field names, then either run [`manager.py`](src/vertex_search_boost/manager.py) directly:

```bash
python3 src/vertex_search_boost/manager.py
```

Or load and sync the bundle from Python (and roll back individual controls when needed):

```python
from vertex_search_boost import VertexBoostManager, load_boost_bundle

# Load and sync the bundled JSON configuration
bundle = load_boost_bundle()
VertexBoostManager.sync_bundle(bundle)

# Roll back: detach a control from ServingConfig (and optionally delete the Control resource)
manager = VertexBoostManager(bundle.target)
manager.detach_control("bury-deprecated-docs", delete_resource=True)
```

---

## How It Works

1. **Piecewise-Linear Interpolation:** Between adjacent control points $(x_i, b_i)$ and $(x_{i+1}, b_{i+1})$, scores follow $b(x) = b_i + \frac{x - x_i}{x_{i+1} - x_i}(b_{i+1} - b_i)$ within $[-1.0, 1.0]$. Values outside the outer anchors clamp flat; documents missing the field evaluate to `0.0` (neutral).
2. **Idempotent Upsert:** `sync_rules()` checks `ControlServiceClient.get_control()` first—updating in place with `FieldMask(paths=["display_name", "boost_action"])` if present, or creating it if absent (with automatic fallback on concurrent `AlreadyExists` races).
3. **Non-Destructive Merge:** Existing `ServingConfig.boost_control_ids` are fetched, merged in order without duplicates, and patched via `FieldMask` unless `replace_existing_controls=True` is explicitly set.

---

## Project Structure

```text
vertex-search-boost/
├── pyproject.toml
├── README.md
├── src/
│   └── vertex_search_boost/
│       ├── __init__.py
│       ├── manager.py                     # Discovery Engine Control & ServingConfig sync
│       ├── models.py                      # Validated rule models, JSON loader & simulator
│       └── multi_signal_boost_bundle.json # Example multi-signal declarative config
└── tests/
    └── test_vertex_search_boost.py        # Offline unit & mock-GCP integration tests
```

---

## Running Unit Tests

All unit tests run offline (using mocks for Discovery Engine gRPC clients) with zero external GCP dependencies:

```bash
pytest
# Or using the standard library test runner:
python3 -m unittest discover -s tests -v
```

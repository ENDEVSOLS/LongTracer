"""
Tests: JSON Schema compatibility and freshness for contracts.

Ensures that:
1. The committed schema docs/schema/case_result.v1.json matches the live CaseResult.
2. No property (top-level or in a nested model) is removed or re-typed, and no
   enum member is removed, unless ``schema_version`` changes.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

from longtracer.contracts.result import CaseResult
from longtracer.contracts.schema import generate_case_result_schema

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "docs" / "schema" / "case_result.v1.json"

# Keys that do not change a field's type or meaning.
_COSMETIC_KEYS = {"title", "description", "default", "examples"}


def _type_signature(spec: Dict[str, Any]) -> Dict[str, Any]:
    """The parts of a property spec that define its type."""
    return {k: v for k, v in spec.items() if k not in _COSMETIC_KEYS}


def _schema_version(schema: Dict[str, Any]) -> str:
    return str(schema["properties"]["schema_version"].get("default"))


def find_breaking_changes(committed: Dict[str, Any], live: Dict[str, Any]) -> List[str]:
    """Return breaking differences from ``committed`` to ``live`` (empty if compatible).

    Breaking: removed property, re-typed property, removed model, removed enum
    member, newly required property. Additive changes are not breaking.
    """
    problems: List[str] = []

    def compare_model(name: str, old: Dict[str, Any], new: Dict[str, Any]) -> None:
        if "enum" in old:
            removed = set(old["enum"]) - set(new.get("enum", []))
            if removed:
                problems.append(f"{name}: enum member(s) removed: {sorted(removed)}")
            return
        old_props, new_props = old.get("properties", {}), new.get("properties", {})
        for prop, spec in old_props.items():
            if prop not in new_props:
                problems.append(f"{name}.{prop}: property removed")
            elif _type_signature(spec) != _type_signature(new_props[prop]):
                problems.append(
                    f"{name}.{prop}: type changed {_type_signature(spec)} -> {_type_signature(new_props[prop])}"
                )
        newly_required = set(new.get("required", [])) - set(old.get("required", []))
        if newly_required:
            problems.append(f"{name}: newly required field(s): {sorted(newly_required)}")

    compare_model("CaseResult", committed, live)
    for def_name, def_spec in committed.get("$defs", {}).items():
        if def_name not in live.get("$defs", {}):
            problems.append(f"$defs.{def_name}: model removed")
        else:
            compare_model(def_name, def_spec, live["$defs"][def_name])
    return problems


def _load_committed() -> Dict[str, Any]:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_schema_file_exists():
    """The committed JSON schema file must exist in docs/schema/."""
    assert SCHEMA_PATH.exists(), f"Schema file missing at {SCHEMA_PATH}"


def test_schema_freshness():
    """Committed JSON schema must match live CaseResult.model_json_schema() exactly."""
    assert _load_committed() == generate_case_result_schema(), (
        "docs/schema/case_result.v1.json is stale. "
        "Run `python -m longtracer.contracts.schema > docs/schema/case_result.v1.json` to regenerate."
    )


def test_schema_has_no_breaking_change_without_version_bump():
    """Removing or re-typing a field requires a schema_version change."""
    committed, live = _load_committed(), generate_case_result_schema()
    if _schema_version(committed) != _schema_version(live):
        return  # version bumped: breaking changes are allowed
    problems = find_breaking_changes(committed, live)
    assert not problems, "Breaking schema change without schema_version bump:\n" + "\n".join(problems)


def test_emitted_schema_version_matches_committed_file():
    assert CaseResult().schema_version == _schema_version(_load_committed()) == "1"


# ── The checker itself must catch each kind of break ─────────────────


def test_checker_detects_removed_property():
    committed = _load_committed()
    live = copy.deepcopy(committed)
    del live["properties"]["quality_gate"]
    assert any("quality_gate: property removed" in p for p in find_breaking_changes(committed, live))


def test_checker_detects_retyped_property():
    committed = _load_committed()
    live = copy.deepcopy(committed)
    live["properties"]["trust_score"] = {"type": "string", "title": "Trust Score"}
    assert any("trust_score: type changed" in p for p in find_breaking_changes(committed, live))


def test_checker_detects_nested_property_removed():
    committed = _load_committed()
    live = copy.deepcopy(committed)
    del live["$defs"]["ClaimResult"]["properties"]["assessment"]
    assert any("ClaimResult.assessment: property removed" in p for p in find_breaking_changes(committed, live))


def test_checker_detects_enum_member_removed():
    committed = _load_committed()
    live = copy.deepcopy(committed)
    live["$defs"]["QualityGate"]["enum"].remove("INDETERMINATE")
    assert any("QualityGate: enum member(s) removed" in p for p in find_breaking_changes(committed, live))


def test_checker_allows_additive_change():
    committed = _load_committed()
    live = copy.deepcopy(committed)
    live["properties"]["new_optional_field"] = {"type": "string", "default": ""}
    live["$defs"]["ReasonCode"]["enum"].append("SOME_NEW_REASON")
    assert find_breaking_changes(committed, live) == []

"""
JSON Schema generator for LongTracer contract models.

Usage:
    python -m longtracer.contracts.schema > docs/schema/case_result.v1.json
"""

from __future__ import annotations

import json
from typing import Any, Dict
from longtracer.contracts.result import CaseResult


def generate_case_result_schema() -> Dict[str, Any]:
    """Generate the JSON schema dictionary for CaseResult."""
    return CaseResult.model_json_schema()


def main() -> None:
    schema = generate_case_result_schema()
    print(json.dumps(schema, indent=2))


if __name__ == "__main__":
    main()
